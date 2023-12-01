from config import cfg
import torchvision.transforms as T


def build_transforms():
    '''
    Image transformations of two types:
        augmentation: data augmentation transforms.
        identity: identity transforms.
        
    Returns:
    ----------
        transforms: dict
            A dictionary of image transformations.
    '''
    resize = T.Resize(cfg.DATA.IMAGE_SIZE)
    random_horizontal_flip = T.RandomHorizontalFlip(p=0.5)
    random_affine = T.RandomAffine(
        degrees=3,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    )
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = {
        'augmentation': T.Compose([resize, random_horizontal_flip, random_affine, to_tensor, normalize]),
        'identity':     T.Compose([resize,                                        to_tensor, normalize]),
    }
    return transforms
