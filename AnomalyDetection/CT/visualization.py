import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from config import cfg
from sklearn.manifold import TSNE
import models
from data import *

def rescale(image, min_val=0, max_val=1):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * (max_val - min_val) + min_val
    return image
 
def train_validation_plots(good_round, bad_round, result_dir):
    with open(f'{result_dir}/train_validation/Good{good_round}_Bad{bad_round}.pickle', 'rb') as f:
        metrics = pickle.load(f)['metrics']
    loss = metrics.get(f'train_loss')
    plt.plot(range(1, 1+len(loss)), loss)
    plt.ylim((0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Training Loss (N{good_round+1} A{bad_round+1})')
    plt.savefig(f'{result_dir}/loss_{good_round+1}{bad_round+1}.jpg', dpi=300)
    plt.show()
    
    validation_auc = metrics.get('validation_auc')
    plt.plot(range(1, 1+len(validation_auc)), validation_auc)
    plt.ylim((0.3, 1))
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'Validation AUC (N{good_round+1} A{bad_round+1})')
    plt.savefig(f'{result_dir}/valauc_{good_round+1}{bad_round+1}.jpg', dpi=300)
    plt.show()

def train_validation_plots_dagmm(good_round, bad_round, result_dir):
    with open(f'{result_dir}/train_validation/Good{good_round}_Bad{bad_round}.pickle', 'rb') as f:
        metrics = pickle.load(f)['metrics']
    plt.plot(range(1, 1+cfg.trainer.num_epochs), metrics.get(f'train_loss'), label='train')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Loss Curves (Good={good_round} Bad={bad_round})')
    # plt.savefig('1.jpg', dpi=300)
    plt.show()
    
    plt.plot(range(1, 1+cfg.trainer.num_epochs), metrics.get('validation_auc'))
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'Validation AUC (Good={good_round} Bad={bad_round})')
    # plt.savefig('1.jpg', dpi=300)
    plt.show()

    plt.plot(range(1, 1+cfg.trainer.num_epochs), metrics.get('recon_error'))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Train Reconstruction Error (Good={good_round} Bad={bad_round})')
    plt.show()

    plt.plot(range(1, 1+cfg.trainer.num_epochs), metrics.get('sample_energy'))
    plt.xlabel('Epoch')
    plt.ylabel('Sample Energy')
    plt.title(f'Train Sample Energy (Good={good_round} Bad={bad_round})')
    plt.show()

    plt.plot(range(1, 1+cfg.trainer.num_epochs), metrics.get('cov_diag'))
    plt.xlabel('Epoch')
    plt.ylabel('Covariance')
    plt.title(f'Train Sample Covariance (Good={good_round} Bad={bad_round})')
    plt.show()

def reconstruct_cae(model, dataloader):
    model.eval()
    images, recons, codes, diffs, rel_dists, cos_sims = [[] for _ in range(6)]
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.cuda()
            code, recon = model(image)
            diff = (image - recon).abs()
            
            image = image[0, 0].detach().cpu().numpy()
            recon = recon[0, 0].detach().cpu().numpy()
            code = code.detach().cpu().numpy().flatten()
            diff = diff[0, 0].detach().cpu().numpy()
            rel_dist = np.linalg.norm(image - recon) / np.linalg.norm(image)
            cos_sim = (image * recon).sum() / (np.linalg.norm(image) * np.linalg.norm(recon))
            
            images.append(image)
            recons.append(recon)
            codes.append(code)
            diffs.append(diff)
            rel_dists.append(rel_dist)
            cos_sims.append(cos_sim)
    anomaly_scores = [(diff**2).mean() for diff in diffs]
    return images, recons, codes, diffs, rel_dists, cos_sims, anomaly_scores

def reconstruct_pcae(model, dataloader):
    model.eval()
    images, recons, diffs, rel_dists, cos_sims = [[] for _ in range(5)]
    for patches, label in dataloader:
            count = np.zeros((cfg.image_channels, cfg.image_size, cfg.image_size))
            image = np.zeros((cfg.image_channels, cfg.image_size, cfg.image_size))
            recon = np.zeros((cfg.image_channels, cfg.image_size, cfg.image_size))

            patches = patches.cuda()
            _, recon_patches = model(patches)

            patches = patches.detach().cpu().numpy()
            recon_patches = recon_patches.detach().cpu().numpy()
            for patch_index in range(patches.shape[0]):
                row_index, col_index = patch_index/cfg.ppd, patch_index%cfg.ppd
                row_index *= cfg.stride
                col_index *= cfg.stride
                count[:, row_index:row_index+cfg.patch_size, col_index:col_index+cfg.patch_size] += 1
                image[:, row_index:row_index+cfg.patch_size, col_index:col_index+cfg.patch_size] += patches[patch_index]
                recon[:, row_index:row_index+cfg.patch_size, col_index:col_index+cfg.patch_size] += recon_patches[patch_index]
            image /= count
            recon /= count
            image = image[0]
            recon = recon[0]
            diff = np.abs(image - recon)
            
            rel_dist = np.linalg.norm(image - recon) / np.linalg.norm(image)
            cos_sim = (image * recon).sum() / (np.linalg.norm(image) * np.linalg.norm(recon))
            
            images.append(image)
            recons.append(recon)
            diffs.append(diff)
            rel_dists.append(rel_dist)
            cos_sims.append(cos_sim)
    anomaly_scores = [(diff**2).mean() for diff in diffs]
    return images, recons, diffs, rel_dists, cos_sims, anomaly_scores

def reconstruct_ssae(model, dataloader):
    model.eval()
    images, recons, diffs = [[] for _ in range(3)]
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.cuda()
            recon = model(image)[0]
            diff = (image - recon).abs()
            
            image = image[0, 0].detach().cpu().numpy()
            recon = recon[0, 0].detach().cpu().numpy()
            diff = diff[0, 0].detach().cpu().numpy()
            
            images.append(image)
            recons.append(recon)
            diffs.append(diff)
    anomaly_scores = [(diff**2).mean() for diff in diffs]
    return images, recons, diffs, anomaly_scores

def reconstruct_dagmm(model, dataloader):
    model.eval()
    images, recons, codes, diffs, rel_dists, cos_sims, anomaly_scores = [[] for _ in range(7)]
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.cuda()
            enc, dec, z, gamma = model(image)
            sample_energy, cov_diag = model.compute_energy(z, size_average=False)

            anomaly_scores.append(sample_energy.item())

            recon = dec.unflatten(1, (1, cfg.image_size, cfg.image_size))
            diff = (image - recon).abs()
            
            image = image[0, 0].detach().cpu().numpy()
            recon = recon[0, 0].detach().cpu().numpy()
            code = enc.detach().cpu().numpy().flatten()
            diff = diff[0, 0].detach().cpu().numpy()
            rel_dist = np.linalg.norm(image - recon) / np.linalg.norm(image)
            cos_sim = (image * recon).sum() / (np.linalg.norm(image) * np.linalg.norm(recon))
            
            images.append(image)
            recons.append(recon)
            codes.append(code)
            diffs.append(diff)
            rel_dists.append(rel_dist)
            cos_sims.append(cos_sim)
    return images, recons, codes, diffs, rel_dists, cos_sims, anomaly_scores

def reconstruct_fanogan(good_folds, bad_folds, good_round, bad_round, result_dir):
    generator = models.fanogan.Generator().cuda()
    discriminator = models.fanogan.Discriminator().cuda()
    encoder = models.fanogan.Encoder().cuda()
    
    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)

    model = torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_GAN.pt')
    generator.load_state_dict(model['generator'])
    discriminator.load_state_dict(model['discriminator'])
    encoder.load_state_dict(torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_encoder.pt'))

    generator.eval()
    discriminator.eval()
    encoder.eval()

    criterion = nn.MSELoss()
    images, recons, diffs, anomaly_scores = [], [], [], []
    with torch.no_grad():
        for real_img, label in dataloaders['test']:
            real_img = real_img.cuda()

            real_z = encoder(real_img)
            fake_img = generator(real_z)

            real_feature = discriminator.forward_features(real_img)
            fake_feature = discriminator.forward_features(fake_img)

            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + cfg.trainer.kappa * loss_feature

            image = real_img[0, 0].detach().cpu().numpy()
            recon = fake_img[0, 0].detach().cpu().numpy()
            diff = np.abs(image - recon)
            images.append(image)
            recons.append(recon)
            diffs.append(diff)
            anomaly_scores.append(anomaly_score.item())

    return images, recons, diffs, anomaly_scores

def anomaly_histogram(anomaly_scores, dataloader):
    num_bad = sum([int(label.item()) for _, label in dataloader])
    plt.hist(anomaly_scores[:-num_bad], bins=20, alpha=0.3, label='Good')
    plt.hist(anomaly_scores[-num_bad:], bins=20, alpha=0.3, label='Bad')
    plt.legend()
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Test Images: Histogram')
    # plt.savefig('1.jpg', dpi=300)
    plt.show()

def cs_reldist(rel_dists, cos_sims, dataloader):
    num_bad = sum([int(label.item()) for _, label in dataloader])
    plt.scatter(rel_dists[:-num_bad], cos_sims[:-num_bad], s=3, label='Good')
    plt.scatter(rel_dists[-num_bad:], cos_sims[-num_bad:], s=3, label='Bad')
    plt.xlabel('Relative Euclidean Distance')
    plt.ylabel('Cosine Similarity')
    plt.title('Test Images: Distance between Original and Reconstruction')
    plt.legend()
    # plt.savefig('1.jpg', dpi=300)
    plt.show()

def tsne_visualization(codes, dataloader):
    num_bad = sum([int(label.item()) for _, label in dataloader])
    codes = np.array(codes)
    embeddings = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca').fit_transform(codes)
    plt.scatter(embeddings[:-num_bad, 0], embeddings[:-num_bad, 1], s=3, label='Good')
    plt.scatter(embeddings[-num_bad:, 0], embeddings[-num_bad:, 1], s=3, label='Bad')
    plt.legend()
    plt.title('Test Images: TSNE Visualization of Latent Embeddings')
    # plt.savefig('1.jpg', dpi=300)
    plt.show()

def anomaly_segmentation(images, recons, diffs, dataloader, i, patch=False):
    num_good = sum([1-int(label.item() if not patch else label[0].item()) for _, label in dataloader])
    
    print(f'Test Image {i} MSE: {(diffs[i]**2).mean()}')

    label = 'Normal' if i < num_good else 'Abnormal'
    
    max_diff = np.max(diffs)
    cmap = 'gray' #if image_channels==1 else None
    image, recon, diff = images[i], recons[i], diffs[i]
    
    diff /= max_diff
    ax1 = plt.subplot(131)
    ax1.imshow(rescale(image), cmap=cmap)
    ax1.axis('off')
    ax1.set_title(f'Test Image: {label}')
    
    ax2 = plt.subplot(132)
    ax2.imshow(rescale(recon), cmap=cmap)
    ax2.axis('off')
    ax2.set_title('Reconstruction')
    
    ax3 = plt.subplot(133)
    ax3.imshow(rescale(diff), cmap=cmap)
    ax3.axis('off')
    ax3.set_title('Absolute Difference')
    
#     plt.savefig(f'{i}.jpg', dpi=300)
    plt.show()
    