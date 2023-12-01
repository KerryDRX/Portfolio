import torch.nn as nn
from torchvision import models
import timm
from config import cfg


def build_model(model_name):
    '''
    Construct a model for training.

    Parameters:
    ----------
        model_name: str
            Name of the model to build.
    
    Returns:
    ----------
        model: torch.nn.Module
            An ImageNet pre-trained model instance.
    '''

    def classification_head(in_features):
        '''
        Build a classification head.

        Parameters:
        ----------
            in_features: int
                Size of input dimension.
        
        Returns:
        ----------
            head: torch.nn.Sequential
                Classification head.
        '''
        head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )
        return head
    
    model_list = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),

        'vgg11': (models.vgg11, models.VGG11_Weights.DEFAULT),
        'vgg11_bn': (models.vgg11_bn, models.VGG11_BN_Weights.DEFAULT),
        'vgg13': (models.vgg13, models.VGG13_Weights.DEFAULT),
        'vgg13_bn': (models.vgg13_bn, models.VGG13_BN_Weights.DEFAULT),
        'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
        'vgg16_bn': (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT),
        'vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT),
        'vgg19_bn': (models.vgg19_bn, models.VGG19_BN_Weights.DEFAULT),
        
        'densenet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        'densenet161': (models.densenet161, models.DenseNet161_Weights.DEFAULT),
        'densenet169': (models.densenet169, models.DenseNet169_Weights.DEFAULT),
        'densenet201': (models.densenet201, models.DenseNet201_Weights.DEFAULT),

        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
        'efficientnet_v2_m': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT),
        'efficientnet_v2_l': (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.DEFAULT),
        
        'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        'vit_b_32': (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
        'vit_l_16': (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT),
        'vit_l_32': (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT),
        'vit_h_14': (models.vit_h_14, models.ViT_H_14_Weights.DEFAULT),
    }
    assert model_name in model_list

    if not model_name.startswith('inception'):
        model_fn, model_weight = model_list[model_name]
        model = model_fn(weights=model_weight)
        if model_name.startswith('resnet'):
            model.fc = classification_head(model.fc.in_features)
        if model_name.startswith('vgg') or model_name.startswith('efficientnet'):
            model.classifier[-1] = classification_head(model.classifier[-1].in_features)
        if model_name.startswith('densenet'):
            model.classifier = classification_head(model.classifier.in_features)
        if model_name.startswith('vit'):
            model.heads.head = classification_head(model.heads.head.in_features)
    else:
        model = timm.create_model(model_name, pretrained=True)
        if model_name == 'inception_v3':
            model.fc = classification_head(model.fc.in_features)
        if model_name == 'inception_v4':
            model.last_linear = classification_head(model.last_linear.in_features)
        if model_name == 'inception_resnet_v2':
            model.classif = classification_head(model.classif.in_features)
    return model
