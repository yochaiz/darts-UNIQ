from os.path import join
import torchvision.datasets as datasets

__DATASETS_DEFAULT_PATH = '/media/ssd/Datasets/'


def get_dataset(name, train, transform, target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    root = datasets_path  # '/mnt/ssd/ImageNet/ILSVRC/Data/CLS-LOC' #os.path.join(datasets_path, name)

    if name == 'cifar10':
        cifar_ = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'cifar100':
        cifar_ = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        return cifar_

    elif name == 'imagenet':
        if train:
            root = join(root, 'train')
        else:
            root = join(root, 'val')

        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)
