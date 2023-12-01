from config import cfg
from utils import *
from data import *
import numpy as np
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F

CAMs = {
    'AblationCAM': AblationCAM,
    # 'DeepFeatureFactorization': DeepFeatureFactorization,
    'EigenCAM': EigenCAM,
    'EigenGradCAM': EigenGradCAM,
    'FullGrad': FullGrad,
    'GradCAM': GradCAM,
    'GradCAMEW': GradCAMElementWise,
    'GradCAMPlusPlus': GradCAMPlusPlus,
    'HiResCAM': HiResCAM,
    'LayerCAM': LayerCAM,
    # 'RandomCAM': RandomCAM,
    'ScoreCAM': ScoreCAM,
    'XGradCAM': XGradCAM,
}

def plots():
    '''
    Make plots for train/validation loss curves and validation/test metrics.
    '''
    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/validation')
    metrics = Logger(f'{cfg.PATHS.OUTPUT_DIR}/metrics.json')
    fig = plt.figure()
    for mode in ['train', 'validation', 'test']:
        loss = metrics.get(f'{mode}_loss')
        plt.plot(range(1, len(loss)+1), loss, label=mode)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.xlim((0, cfg.TRAINER.NUM_EPOCHS)); plt.legend(); plt.title('Loss Curves')
    fig.savefig(f'{cfg.PATHS.OUTPUT_DIR}/validation/losses.jpg', dpi=300)
    plt.close('all')

    for metric_name in metric_functions:
        fig = plt.figure()
        for mode in ['validation', 'test']:
            metric = metrics.get(f'{mode}_{metric_name}')
            plt.plot(range(1, len(metric)+1), metric, label=mode)
        title = metric_name.capitalize() if metric_name != 'auc' else 'AUC'
        plt.xlabel('Epoch'); plt.ylabel(title); plt.xlim((0, cfg.TRAINER.NUM_EPOCHS)); plt.ylim((0, 1))
        plt.legend(loc=4); plt.title(f'Validation and Test {title}')
        fig.savefig(f'{cfg.PATHS.OUTPUT_DIR}/validation/validation_{metric_name}.jpg', dpi=300)
        plt.close('all')
        
def gradcam_visualization(model, dataloaders, preds, mode):
    '''
    GradCAM visualization.

    Parameters:
    ----------
        model: torch.nn.Module
            The model to test.
        dataloaders: dict
            Dictionary of dataloaders.
        preds: torch.Tensor
            Predicted labels.
    '''
    
    def _get_visualization(CAM_name, model, image, np_image):
        '''
        GradCAM visualization of an image.

        Parameters:
        ----------
            CAM_name: str
                The type of GradCAM to use.
            model: torch.nn.Module
                The model to test.
            image: torch.Tensor
                Image tensor with shape [1, C, H, W].
            np_image: numpy.ndarray
                Image array with shape [H, W, C].
            
        Returns:
        ----------
            visualization0: numpy.ndarray
                Heatmap with shape [H, W, C] for the negative (good) class.
            visualization1: numpy.ndarray
                Heatmap with shape [H, W, C] for the positive (bad) class.
        '''
        with CAMs[CAM_name](
            model=model, 
            target_layers=[model.features[-1]] if CAM_name != 'FullGrad' else [], 
            use_cuda=True
        ) as cam:
            visualization0, visualization1 = [
                show_cam_on_image(
                    np_image,
                    cam(
                        input_tensor=image, 
                        targets=[ClassifierOutputSoftmaxTarget(category)], 
                        aug_smooth=(CAM_name != 'HiResCAM'), 
                        eigen_smooth=(CAM_name != 'HiResCAM'),
                    )[0], 
                    use_rgb=True
                )
                for category in [0, 1]
            ]
        return visualization0, visualization1

    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/{mode}')
    i = 0
    for (image, label), pred in zip(dataloaders[mode], preds):
        if i+1 < 75:
            i += 1
            continue
        print(f'\n{mode} image {i+1}/{len(dataloaders[mode])}\n')
        label = cfg.DATA.GOOD_LABEL if label == 0 else cfg.DATA.BAD_LABEL
        pred = cfg.DATA.GOOD_LABEL if pred.item() == 0 else cfg.DATA.BAD_LABEL
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logit = model(image.cuda())
                prob = F.softmax(logit, dim=1).max().item()

        np_image = np.transpose(image[0].numpy(), (1,2,0))
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
        visualizations = {CAM_name: _get_visualization(CAM_name, model, image, np_image) for CAM_name in CAMs}
        fig = plt.figure(constrained_layout=True, figsize=(2+len(visualizations), 2))
        gs = GridSpec(2, 2+len(visualizations), figure=fig)
        ax = fig.add_subplot(gs[:, :2])
        ax.imshow(np_image)
        ax.axis('off')
        ax.set_title(f'GT={label}\nPred={pred} (prob={prob:.2f})', fontsize=10)
        for row in range(2):
            for CAM_name, col in zip(visualizations, range(2, 13)):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(visualizations[CAM_name][row])
                ax.axis('off')
                if row == 0:
                    ax.set_title(CAM_name, fontsize=7)
        
        if label == pred == cfg.DATA.GOOD_LABEL:
            result = 'TN'
        if label == pred == cfg.DATA.BAD_LABEL:
            result = 'TP'
        if label == cfg.DATA.GOOD_LABEL and pred == cfg.DATA.BAD_LABEL:
            result = 'FP'
        if label == cfg.DATA.BAD_LABEL and pred == cfg.DATA.GOOD_LABEL:
            result = 'FN'
        plt.savefig(f'{cfg.PATHS.OUTPUT_DIR}/{mode}/{result}_{prob:.2f}_{i}.jpg', dpi=1500)
        plt.close('all')
        i += 1
        