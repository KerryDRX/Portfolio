from torchvision.models import resnet18


def get_model(dataset_name):
    if dataset_name == 'CIFAR10':
        return resnet18(num_classes=10)
    else:
        raise NotImplementedError
