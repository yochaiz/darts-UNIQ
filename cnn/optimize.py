from time import time

from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import no_grad
from torch.optim import SGD
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss

from cnn.utils import accuracy, AvgrageMeter, load_data
from cnn.utils import initTrainLogger, logDominantQuantizedOp, save_checkpoint
from cnn.architect import Architect


def train(train_queue, search_queue, args, model, architect, crit, optimizer, lr, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    grad = AvgrageMeter()

    model.train()
    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        if len(search_queue) > 0:
            input_search, target_search = next(iter(search_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda(async=True)

            arch_grad_norm = architect.step(input, target, input_search, target_search, lr,
                                            optimizer, unrolled=args.unrolled)
            grad.update(arch_grad_norm)

        optimizer.zero_grad()
        logits = model(input)
        loss = crit(logits, target)

        loss.backward()
        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()

        if step % args.report_freq == 0:
            logger.info('train [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] BopsRatio:[{:.3f}] BopsLoss[{:.5f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, model._criterion.bopsRatio,
                                model._criterion.quant_loss, endTime - startTime))

    return top1.avg, objs.avg


def infer(valid_queue, args, model, crit, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.eval()
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

            if step % args.report_freq == 0:
                logger.info(
                    'validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] BopsRatio:[{:.3f}] BopsLoss[{:.5f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, model._criterion.bopsRatio,
                                model._criterion.quant_loss, endTime - startTime))

    return top1.avg, objs.avg


def optimize(args, model, logger):
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
    nEpochs = epochsSwitchStage[-1] + 1
    # remove epoch 0 from list, we don't want to switch stage at the beginning
    epochsSwitchStage = epochsSwitchStage[1:]

    logger.info('nEpochs:[{}]'.format(nEpochs))
    logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

    scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
    architect = Architect(model, args)

    best_prec1 = 0.0

    for epoch in range(1, nEpochs + 1):
        trainLogger = initTrainLogger(str(epoch), trainFolderPath, args.propagate)

        scheduler.step()
        lr = scheduler.get_lr()[0]

        trainLogger.info('optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.defaults['lr'], lr))

        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_loss = train(train_queue, search_queue, args, model, architect, cross_entropy,
                                      optimizer, lr, trainLogger)

        # log accuracy, loss, etc.
        message = 'Epoch:[{}] , training accuracy:[{:.3f}] , training loss:[{:.3f}] , optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]' \
            .format(epoch, train_acc, train_loss, optimizer.defaults['lr'], lr)
        logger.info(message)
        trainLogger.info(message)

        # log dominant QuantizedOp in each layer
        logDominantQuantizedOp(model, k=3, logger=trainLogger)

        # save model checkpoint
        save_checkpoint(trainFolderPath, model, epoch, best_prec1, is_best=False)

        # switch stage, i.e. freeze one more layer
        if (epoch in epochsSwitchStage) or (epoch == nEpochs):
            # validation
            valid_acc, valid_loss = infer(valid_queue, args, model, cross_entropy, trainLogger)
            message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}]'.format(epoch, valid_acc,
                                                                                                    valid_loss)
            logger.info(message)
            trainLogger.info(message)

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(trainFolderPath, model, epoch, best_prec1, is_best)

            # switch stage
            model.switch_stage(trainLogger)
            # update optimizer & scheduler due to update in learnable params
            optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                            momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
