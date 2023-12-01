from torchvision import transforms as T


default_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

augmentations = {
    'CIFAR10': T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.RandomRotation(15)]),
    'SVHN': T.RandomCrop(32, padding=4),
}

transforms = {
    dataset_name: {
        'augmentation': T.Compose([augmentation, default_transforms]),
        'identity': default_transforms,
    } for dataset_name, augmentation in augmentations.items()
}
