import torch
from pytorch_gan_metrics import get_inception_score_and_fid
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from transforms import channel_2to3
from data import *
from model import *
from config import cfg
from utils import *


def evaluate_generator(generator, fid_stats_path):
    '''
    Evaluate generator performance by comparing the generated images and the original training images.

    Parameters:
    ----------
        generator: Generator
            The generator to evaluate.
        fid_stats_path: str
            The file path that stores training image statistics.
    
    Returns:
    ----------
        IS: tuple
            Mean and std of inception score.
        FID: float
            Fréchet inception distance.
    '''
    generator.eval()
    images = []
    num_images = 5000
    with torch.no_grad():
        for start in range(0, num_images, cfg.TRAINER.BATCH_SIZE):
            end = min(start + cfg.TRAINER.BATCH_SIZE, num_images)
            z = torch.randn(end - start, cfg.DATA.LATENT_DIM).cuda()
            image = generator(z).cpu()
            images.append(image)
    images = torch.cat(images)
    images = channel_2to3(image=images)['image']
    IS, FID = get_inception_score_and_fid(images, fid_stats_path, verbose=False)
    return IS, FID

def evaluate_encoder(generator, discriminator, encoder, dataloader):
    '''
    Evaluate encoder performance by calculating the AUC score of the entire f-AnoGAN model.

    Parameters:
    ----------
        generator: Generator
            The generator of f-AnoGAN.
        discriminator: Discriminator
            The discriminator of f-AnoGAN.
        encoder: Discriminator
            The encoder of f-AnoGAN.
        dataloader: torch.utils.data.dataloader.DataLoader
            The dataloader to test on.
    
    Returns:
    ----------
        auc: float
            The AUC on the test dataset.
        labels: list
            Test image labels.
        anomaly_scores: list
            Anomaly scores of the test images.
        real_images: list
            Original test images.
        fake_images: list
            Reconstructed test images.
    '''
    generator.eval(); discriminator.eval(); encoder.eval()
    criterion = torch.nn.MSELoss()
    labels, anomaly_scores, real_images, fake_images = [], [], [], []
    with torch.no_grad():
        for real_image, label in dataloader:
            real_image = real_image.cuda()
            fake_image = generator(encoder(real_image))
            real_feature = discriminator.extract_features(real_image)
            fake_feature = discriminator.extract_features(fake_image)
            anomaly_score = criterion(fake_image, real_image) + cfg.TRAINER.ENCODER.KAPPA * criterion(fake_feature, real_feature)
            real_images.append(real_image.detach().cpu().numpy())
            fake_images.append(fake_image.detach().cpu().numpy())
            labels.append(label.item())
            anomaly_scores.append(anomaly_score.item())
    auc = roc_auc_score(labels, anomaly_scores) if np.unique(labels).size == 2 else 0
    return auc, labels, anomaly_scores, real_images, fake_images

def test():
    '''
    Test f-AnoGAN performance on train/validation/test sets and visualize all the results.
    Calculate AUC score, plot anomaly histogram and difference maps.
    '''

    # load trained f-AnoGAN model
    checkpoint_gan = torch.load(f'{cfg.PATHS.OUTPUT_DIR}/model_gan/final.pt')
    checkpoint_encoder = torch.load(f'{cfg.PATHS.OUTPUT_DIR}/model_encoder/best.pt')
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    encoder = Encoder().cuda()
    generator.load_state_dict(checkpoint_gan['generator'])
    discriminator.load_state_dict(checkpoint_gan['discriminator'])
    encoder.load_state_dict(checkpoint_encoder['encoder'])
    generator.eval()
    discriminator.eval()
    encoder.eval()
    enable_grad(generator, False)
    enable_grad(discriminator, False)
    enable_grad(encoder, False)

    def _to_rgb(image):
        '''
        Given an image, add blue channel, rescale to [0, 1], and change axes order to [H, W, C].

        Parameters:
        ----------
            image: numpy.ndarray
                Image array with shape [2, H, W].
        
        Returns:
        ----------
            image_rgb: numpy.ndarray
                RGB image with shape [H, W, 3].
        '''
        image_rgb = np.zeros((3, *cfg.DATA.IMAGE_SIZE))
        image_rgb[:2] = (image[0] + 1) / 2
        image_rgb = np.transpose(image_rgb, (1, 2, 0))
        return image_rgb

    # model testing
    dataloaders = build_dataloaders()
    for eval_mode in ['train_identity2', 'validation', 'test']:
        mkdir(f'{cfg.PATHS.OUTPUT_DIR}/recons_{eval_mode}')

        # anomaly histogram visualization
        auc, labels, anomaly_scores, real_images, fake_images = evaluate_encoder(generator, discriminator, encoder, dataloaders[eval_mode])
        plt.hist([anomaly_score for i, anomaly_score in enumerate(anomaly_scores) if labels[i] == 0], alpha=0.3, bins=100, label=cfg.DATA.GOOD_LABEL)
        plt.hist([anomaly_score for i, anomaly_score in enumerate(anomaly_scores) if labels[i] == 1], alpha=0.3, bins=100, label=cfg.DATA.BAD_LABEL)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.xlim((0, 0.2))
        plt.title(f'{eval_mode.capitalize()} Histogram (AUC {auc:.3f})' if auc > 0 else 'Train Histogram')
        plt.legend()
        plt.savefig(f'{cfg.PATHS.OUTPUT_DIR}/recons_{eval_mode}/anomaly_histogram.jpg', dpi=300)
        plt.show()
        plt.close('all')

        # image reconstruction and difference map visualization
        for idx, (image, recon, label, anomaly_score) in enumerate(zip(real_images, fake_images, labels, anomaly_scores)):
            image, recon = _to_rgb(image), _to_rgb(recon)
            diff = np.abs(image - recon)
            _, axes = plt.subplots(1, 3)
            axes[0].imshow(image)
            label = cfg.DATA.GOOD_LABEL if label == 0 else cfg.DATA.BAD_LABEL
            axes[0].set_title(f'Ground Truth: {label}')
            axes[1].imshow(recon)
            axes[1].set_title(f'Score: {anomaly_score:.3f}')
            axes[2].imshow(diff)
            axes[2].set_title(f'Difference')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'{cfg.PATHS.OUTPUT_DIR}/recons_{eval_mode}/{label}_{anomaly_score:.3f}_{idx}.jpg', dpi=300)
            plt.show()
            plt.close('all')
