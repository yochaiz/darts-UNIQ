from time import time

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import no_grad
from torch.optim import SGD
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss

from cnn.utils import accuracy, AvgrageMeter, load_data
from cnn.utils import initTrainLogger, logDominantQuantizedOp, save_checkpoint
from cnn.architect import Architect
from cnn.model_replicator import ModelReplicator
from cnn.statistics import Statistics


def trainWeights(train_queue, search_queue, args, model, crit, optimizer, lr, logger):
    loss_container = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.train()
    model.trainMode()

    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # choose optimal alpha per layer
        bopsRatio = model.chooseRandomPath()
        # optimize model weights
        optimizer.zero_grad()
        logits = model(input)
        loss = crit(logits, target)
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        # print('w grads:{}'.format(model.block1.alphas.grad))
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        loss_container.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()
        if step % args.report_freq == 0:
            logger.info(
                'train [{}/{}] weight_loss:[{:.5f}] Accuracy:[{:.3f}] BopsRatio:[{:.3f}] time:[{:.5f}]'
                    .format(step, nBatches, loss_container.avg, top1.avg, bopsRatio, endTime - startTime))

    return top1.avg, loss_container.avg


def trainAlphas(search_queue, model, architect, stats, nEpoch, logger):
    loss_container = AvgrageMeter()

    model.train()

    nBatches = len(search_queue)

    for step, (input, target) in enumerate(search_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        model.trainMode()
        loss = architect.step(model, input, target)

        # add alphas data to statistics
        stats.addBatchData(model, nEpoch, step)
        # log dominant QuantizedOp in each layer
        logDominantQuantizedOp(model, k=3, logger=logger)
        # save alphas to csv
        model.save_alphas_to_csv(data=[nEpoch, step])
        # save loss to container
        loss_container.update(loss, n)
        # count current optimal model bops
        bopsRatio = model.evalMode()

        endTime = time()
        logger.info('train [{}/{}] arch_loss:[{:.5f}] BopsRatio:[{:.3f}] time:[{:.5f}]'
                    .format(step, nBatches, loss_container.avg, bopsRatio, endTime - startTime))

    return loss_container.avg


def infer(valid_queue, model, crit, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.eval()
    bopsRatio = model.evalMode()
    # print eval layer index selection
    logger.info('Layers optimal indices:{}'.format([layer.curr_alpha_idx for layer in model.layersList]))

    nBatches = len(valid_queue)

    with no_grad():
        for step, (input, target) in enumerate(valid_queue):
            startTime = time()

            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)

            logits = model(input)
            loss = crit(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            endTime = time()

            logger.info('validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] BopsRatio:[{:.3f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, bopsRatio, endTime - startTime))

    return top1.avg, objs.avg, bopsRatio


def optimize(args, model, uniform_model, modelClass, logger):
    trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

    optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = Adam(model.parameters(), lr=args.learning_rate,
    #                  betas=(0.5, 0.999), weight_decay=args.weight_decay)

    # init cross entropy loss
    cross_entropy = CrossEntropyLoss().cuda()

    # load data
    train_queue, search_queue, valid_queue = load_data(args)

    # extend epochs list as number of model layers
    while len(args.epochs) < model.nLayers():
        args.epochs.append(args.epochs[-1])
    # init epochs number where we have to switch stage in
    epochsSwitchStage = [0]
    for e in args.epochs:
        epochsSwitchStage.append(e + epochsSwitchStage[-1])
    # total number of epochs is the last value in epochsSwitchStage
    nEpochs = epochsSwitchStage[-1] + args.epochs[-1]
    # remove epoch 0 from list, we don't want to switch stage at the beginning
    epochsSwitchStage = epochsSwitchStage[1:]

    logger.info('nEpochs:[{}]'.format(nEpochs))
    logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

    scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

    # init validation best precision value
    best_prec1 = 0.0
    # init epoch
    epoch = 0

    for epoch in range(1, nEpochs + 1):
        trainLogger = initTrainLogger(str(epoch), trainFolderPath, args.propagate)
        is_best = False

        scheduler.step()
        lr = scheduler.get_lr()[0]

        trainLogger.info('optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.defaults['lr'], lr))

        # training
        print('========== Epoch:[{}] =============='.format(epoch))
        train_acc, train_loss = trainWeights(train_queue, search_queue, args, model, cross_entropy,
                                             optimizer, lr, trainLogger)

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , training accuracy:[{:.3f}] , training loss:[{:.3f}] , optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]' \
            .format(epoch, train_acc, train_loss, optimizer.defaults['lr'], lr)
        logger.info(message)
        trainLogger.info(message)

        # log dominant QuantizedOp in each layer
        logDominantQuantizedOp(model, k=3, logger=trainLogger)

        # switch stage, i.e. freeze one more layer
        if (epoch in epochsSwitchStage) or (epoch == nEpochs):
            # validation
            valid_acc, valid_loss, bopsRatio = infer(valid_queue, model, cross_entropy, trainLogger)
            message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}] , BopsRatio:[{:.3f}]' \
                .format(epoch, valid_acc, valid_loss, bopsRatio)
            logger.info(message)
            trainLogger.info(message)

            # # update values for opt model decision
            # is_best = valid_acc > best_prec1
            # best_prec1 = max(valid_acc, best_prec1)

            # switch stage
            model.switch_stage(trainLogger)
            # update optimizer & scheduler due to update in learnable params
            optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                            momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

        # save model checkpoint
        save_checkpoint(trainFolderPath, model, epoch, best_prec1, is_best=False)

    # init model replicator object
    modelReplicator = ModelReplicator(model, modelClass, args)
    # init statistics instance
    stats = Statistics(model.layersList, model.nLayers(), args.save)
    # init architect
    architect = Architect(modelReplicator, args)
    # turn on alphas
    model.turnOnAlphas()
    #copy weights to uniform model
    modelStateDict = model.state_dict()
    uniform_model.loadBitsWeigths(modelStateDict,args.MaxBopsBits,args.bitwidth)
    # init scheduler
    nEpochs = model.nLayers()
    scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
    # init validation best precision value
    best_prec1 = 0.0
    # train alphas
    for epoch in range(epoch + 1, epoch + nEpochs + 1):
        trainLogger = initTrainLogger(str(epoch), trainFolderPath, args.propagate)

        scheduler.step()
        lr = scheduler.get_lr()[0]

        trainLogger.info('optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.defaults['lr'], lr))
        print('========== Epoch:[{}] =============='.format(epoch))
        arch_loss = trainAlphas(search_queue, model, architect, stats, epoch, trainLogger)

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , arch loss:[{:.3f}] , optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]' \
            .format(epoch, arch_loss, optimizer.defaults['lr'], lr)
        logger.info(message)
        trainLogger.info(message)

        # validation model
        valid_acc, valid_loss, bopsRatio = infer(valid_queue, model, cross_entropy, trainLogger)
        message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}] , BopsRatio:[{:.3f}]' \
            .format(epoch, valid_acc, valid_loss, bopsRatio)

        logger.info(message)
        trainLogger.info(message)


        # validation uniform model
        trainLogger.info('== Validation uniform model ==')
        valid_uni_acc, valid_uni_loss, _ = infer(valid_queue, uniform_model, cross_entropy, trainLogger)
        message = 'Epoch:[{}] , uniform validation accuracy:[{:.3f}] , uniform validation loss:[{:.3f}]]' \
            .format(epoch, valid_uni_acc, valid_uni_loss)

        logger.info(message)
        trainLogger.info(message)

        # save model checkpoint
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        save_checkpoint(trainFolderPath, model, epoch, best_prec1, is_best)

# def train(train_queue, search_queue, args, model, architect, crit, optimizer, lr, logger):
#     weights_loss_container = AvgrageMeter()
#     arch_loss_container = AvgrageMeter()
#     top1 = AvgrageMeter()
#     top5 = AvgrageMeter()
#     grad = AvgrageMeter()
#
#     model.train()
#
#     nBatches = len(train_queue)
#
#     for step, (input, target) in enumerate(train_queue):
#         startTime = time()
#         n = input.size(0)
#
#         input = Variable(input, requires_grad=False).cuda()
#         target = Variable(target, requires_grad=False).cuda(async=True)
#
#         # get a random minibatch from the search queue with replacement
#         if (len(search_queue) > 0) and (len(model.arch_parameters()) > 0):
#             input_search, target_search = next(iter(search_queue))
#             input_search = Variable(input_search, requires_grad=False).cuda()
#             target_search = Variable(target_search, requires_grad=False).cuda(async=True)
#
#             arch_grad_norm, arch_loss = architect.step(input, target, input_search, target_search, lr,
#                                                        optimizer, unrolled=args.unrolled)
#             grad.update(arch_grad_norm)
#
#         # choose optimal alpha per layer
#         bopsRatio = model.trainMode()
#         # optimize model weights
#         optimizer.zero_grad()
#         logits = model(input)
#         loss = crit(logits, target)
#         loss.backward()
#         clip_grad_norm_(model.parameters(), args.grad_clip)
#         # print('w grads:{}'.format(model.block1.alphas.grad))
#         optimizer.step()
#
#         # normalize alphas
#         # for alphas in model.arch_parameters():
#         #     minNorm = abs(alphas).min()
#         #     alphas.data = tensor((alphas / minNorm).cuda(), requires_grad=True)
#
#         prec1, prec5 = accuracy(logits, target, topk=(1, 5))
#         weights_loss_container.update(loss.item(), n)
#         arch_loss_container.update(arch_loss.item(), len(search_queue))
#         top1.update(prec1.item(), n)
#         top5.update(prec5.item(), n)
#
#         endTime = time()
#         if step % args.report_freq == 0:
#             logger.info(
#                 'train [{}/{}] weight_loss:[{:.5f}] Accuracy:[{:.3f}] arch_loss:[{:.5f}] BopsRatio:[{:.3f}] time:[{:.5f}]'
#                     .format(step, nBatches, weights_loss_container.avg, top1.avg, arch_loss_container.avg,
#                             bopsRatio, endTime - startTime))
#
#     return top1.avg, weights_loss_container.avg, arch_loss_container.avg
