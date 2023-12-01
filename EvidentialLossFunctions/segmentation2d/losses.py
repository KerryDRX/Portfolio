import monai
import torch


# class DiceLoss(torch.nn.Module):
#     def __init__(self, soft, smooth):
#         super(DiceLoss, self).__init__()
#         self.soft = soft
#         self.smooth = smooth

#     def forward(self, input, target):
#         input = torch.sigmoid(input)  # (B, 1, H, W)
#         reduce_axis = torch.arange(2, len(input.shape)).tolist()  # (H, W)
#         intersection = torch.sum(target * input, dim=reduce_axis)
#         ground_o = torch.sum(target**2 if self.soft else target, dim=reduce_axis)
#         pred_o = torch.sum(input**2 if self.soft else input, dim=reduce_axis)
#         denominator = ground_o + pred_o
#         score = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
#         return torch.mean(score)


# class FocalTverskyLoss(torch.nn.Module):
#     def __init__(self, alpha, beta, gamma, smooth):
#         super(FocalTverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.smooth = smooth

#     def forward(self, input, target):
#         input = torch.sigmoid(input)  # (B, 1, H, W)
#         reduce_axis = torch.arange(2, len(input.shape)).tolist()  # (H, W)
#         tp = torch.sum(input * target, reduce_axis)
#         fp = torch.sum(input * (1 - target), reduce_axis)
#         fn = torch.sum((1 - input) * target, reduce_axis)
#         numerator = tp + self.smooth
#         denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
#         score = (1.0 - numerator / denominator).pow(self.gamma)  # (B, 1)
#         return torch.mean(score)

loss_fns = {
    'CE':           monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=0.0, lambda_dice=0.0, lambda_focal=1.0),
    'Focal_gamma1': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=1.0, lambda_dice=0.0, lambda_focal=1.0),
    'Focal_gamma2': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=2.0, lambda_dice=0.0, lambda_focal=1.0),
    'Focal_gamma3': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=3.0, lambda_dice=0.0, lambda_focal=1.0),

    'Dice':  monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=0.0, lambda_dice=1.0, lambda_focal=0.0),
    'SDice': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=0.0, lambda_dice=1.0, lambda_focal=0.0),
    
    'DiceCE':  monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=0.0, lambda_dice=1.0, lambda_focal=1.0),
    'SDiceCE': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=0.0, lambda_dice=1.0, lambda_focal=1.0),
    
    'DiceFocal_gamma2':  monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=False, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0),
    'SDiceFocal_gamma2': monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0),
}

# loss_fns = {
#     'Dice_smooth':         DiceLoss(soft=False, smooth=1),
#     'SDice_smooth':        DiceLoss(soft=True, smooth=1),
#     'Tversky_smooth':      FocalTverskyLoss(alpha=0.3, beta=0.7, smooth=1, gamma=1),
#     'FocalTversky_smooth': FocalTverskyLoss(alpha=0.3, beta=0.7, smooth=1, gamma=0.75),
# }
