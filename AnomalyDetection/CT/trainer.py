import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score
from data import *
from utils import *
import pickle
from tqdm import tqdm, trange
import models
from config import cfg


def train_cae(good_folds, bad_folds, good_round, bad_round, result_dir, patch=False):
    mkdir(f'{result_dir}/best_models')
    mkdir(f'{result_dir}/final_models')
    mkdir(f'{result_dir}/train_validation')
    mkdir(f'{result_dir}/test')

    model = models.cae.AutoEncoder(patch).cuda()
    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round, patch)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr, betas=(cfg.trainer.b1, cfg.trainer.b2))
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    metrics = Metrics()
    best_auc = 0
    validate = validate_cae if not patch else validate_pcae
    with trange(1, cfg.trainer.num_epochs+1, desc=f'{good_round}_{bad_round}') as pbar:
        for epoch in pbar:
            model.train()
            losses = []
            for image, _ in dataloaders['train']:
                image = image.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    _, recon = model(image)
                    loss = criterion(recon, image)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            train_loss = np.mean(losses)
            metrics.log({
                'train_loss': train_loss,
            })
            if epoch % cfg.trainer.val_interval == 0:
                validation_loss, validation_auc, validation_labels, validation_anomaly_scores = validate(model, dataloaders['validation'])
                metrics.log({
                    'validation_loss': validation_loss,
                    'validation_auc': validation_auc,
                })

                if validation_auc > best_auc:
                    best_auc = validation_auc
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt')
                    metrics.log({
                        'best_validation_labels': validation_labels,
                        'best_validation_anomaly_scores': validation_anomaly_scores,
                    })
                pbar.set_postfix(AUC=f'{validation_auc:.4f} (best={best_auc:.4f} epoch={best_epoch})')

            torch.save(model.state_dict(), f'{result_dir}/final_models/Good{good_round}_Bad{bad_round}.pt')
    
    with open(f'{result_dir}/train_validation/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
        pickle.dump({'metrics': metrics, 'best_auc': best_auc}, f)

def train_ssae(good_folds, bad_folds, good_round, bad_round, result_dir):
    mkdir(f'{result_dir}/best_models')
    mkdir(f'{result_dir}/final_models')
    mkdir(f'{result_dir}/train_validation')
    mkdir(f'{result_dir}/test')

    model = models.ssae.ScaleSpaceAutoEncoder().cuda()
    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr, betas=(cfg.trainer.b1, cfg.trainer.b2))
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    metrics = Metrics()
    best_auc = 0
    with trange(1, cfg.trainer.num_epochs+1, desc=f'{good_round}_{bad_round}') as pbar:
        for epoch in pbar:
            model.train()
            losses = []
            for image0, _ in dataloaders['train']:
                image0 = image0.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    recon0, recon1, recon2, down_image1, down_image2 = model(image0)
                    loss = cfg.trainer.lambda0 * criterion(image0, recon0) + cfg.trainer.lambda1 * criterion(down_image1, recon1) + cfg.trainer.lambda2 * criterion(down_image2, recon2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            train_loss = np.mean(losses)
            metrics.log({
                'train_loss': train_loss,
            })
            if epoch % cfg.trainer.val_interval == 0:
                validation_auc, validation_labels, validation_anomaly_scores = validate_ssae(model, dataloaders['validation'])
                metrics.log({
                    'validation_auc': validation_auc,
                })

                if validation_auc > best_auc:
                    best_auc = validation_auc
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt')
                    metrics.log({
                        'best_validation_labels': validation_labels,
                        'best_validation_anomaly_scores': validation_anomaly_scores,
                    })
                pbar.set_postfix(AUC=f'{validation_auc:.4f} (best={best_auc:.4f} epoch={best_epoch})')

            torch.save(model.state_dict(), f'{result_dir}/final_models/Good{good_round}_Bad{bad_round}.pt')
    
    with open(f'{result_dir}/train_validation/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
        pickle.dump({'metrics': metrics, 'best_auc': best_auc}, f)

def train_dagmm(good_folds, bad_folds, good_round, bad_round, result_dir):
    mkdir(f'{result_dir}/best_models')
    mkdir(f'{result_dir}/final_models')
    mkdir(f'{result_dir}/train_validation')
    mkdir(f'{result_dir}/test')

    model = models.dagmm.DAGMM().cuda()
    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    scaler = torch.cuda.amp.GradScaler()
    metrics = Metrics()
    best_auc = 0
    with trange(1, cfg.trainer.num_epochs+1, desc=f'{good_round}_{bad_round}') as pbar:
        for epoch in pbar:
            model.train()
            losses = []
            for image, _ in dataloaders['train']:
                image = image.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    enc, dec, z, gamma = model(image)
                    loss, sample_energy, recon_error, cov_diag = model.loss_function(image, dec, z, gamma, cfg.trainer.lambda_energy, cfg.trainer.lambda_cov_diag)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            train_loss = np.mean(losses)
            validation_auc = validate_dagmm(model, dataloaders)
            metrics.log({
                'train_loss': train_loss,
                'validation_auc': validation_auc,
                'recon_error': recon_error.item(),
                'sample_energy': sample_energy.item(),
                'cov_diag': cov_diag.item(),
                # 'validation_loss': validation_loss,
            })
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_epoch = epoch
                torch.save(model, f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt')
            torch.save(model, f'{result_dir}/final_models/Good{good_round}_Bad{bad_round}.pt')

            pbar.set_postfix(AUC=f'{validation_auc:.4f} (best={best_auc:.4f} epoch={best_epoch})')
    
    with open(f'{result_dir}/train_validation/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
        pickle.dump({'metrics': metrics, 'best_auc': best_auc}, f)

def train_wgangp(good_folds, bad_folds, good_round, bad_round, result_dir):
    mkdir(f'{result_dir}/best_models')
    mkdir(f'{result_dir}/final_models')
    mkdir(f'{result_dir}/train_validation')
    mkdir(f'{result_dir}/test')
    mkdir(f'{result_dir}/images')

    generator = models.fanogan.Generator().cuda()
    discriminator = models.fanogan.Discriminator().cuda()

    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)
    calc_and_save_stats(dataloaders['train_orig'])

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=cfg.trainer.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.trainer.lr)
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.GAN.lr, betas=(cfg.GAN.b1, cfg.GAN.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.GAN.lr, betas=(cfg.GAN.b1, cfg.GAN.b2))
    # scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lambda step: 1 - step / 10000)
    # scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda step: 1 - step / 10000)

    batches_done = 0
    metrics = Metrics()
    best_FID = np.Inf
    with trange(1, cfg.trainer.num_epochs+1, desc=f'{good_round}_{bad_round}') as pbar:
        for epoch in pbar:
            generator.train()
            discriminator.train()

            for i, (real_imgs, _) in enumerate(dataloaders['train']):
                real_imgs = real_imgs.cuda()

                optimizer_D.zero_grad()
                z = torch.randn(real_imgs.shape[0], cfg.latent_dim).cuda()
                fake_imgs = generator(z)
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                d_loss = fake_validity.mean() - real_validity.mean() + cfg.trainer.lambda_gp * gradient_penalty  # Adversarial loss
                d_loss.backward()
                optimizer_D.step()
                metrics.log({'d_loss': d_loss.item()})

                optimizer_G.zero_grad()
            
            if epoch % cfg.trainer.n_critic == 0:
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -fake_validity.mean()
                g_loss.backward()
                optimizer_G.step()
                metrics.log({'g_loss': g_loss.item()})

                log(
                    f"[Epoch {epoch:{len(str(cfg.trainer.num_epochs))}}/{cfg.trainer.num_epochs}] "
                    f"[Batch {i+1:{len(str(len(dataloaders['train'])))}}/{len(dataloaders['train'])}] "
                    f"[D loss: {d_loss.item():3f}] "
                    f"[G loss: {g_loss.item():3f}]",
                    pt=False,
                )
                # pbar.set_postfix(Loss=f'D={d_loss.item():.3f} G={g_loss.item():.3f}')
                if batches_done % cfg.trainer.save_image_interval == 0:
                    save_image(
                        fake_imgs.data[:25],
                        f'{result_dir}/images/{batches_done:06}.png',
                        nrow=5, 
                        normalize=True,
                    )
                batches_done += cfg.trainer.n_critic

            if epoch % cfg.trainer.save_model_interval == 0:
                IS, FID = evaluate_generator(generator)
                log(f'InceptionScore={IS[0]:.3f}({IS[1]:.3f}) FID={FID:.3f}', pt=False)
                if FID <= best_FID:
                    best_FID = FID
                    best_epoch = epoch
                    torch.save(
                        {
                            'generator': generator.state_dict(),
                            'discriminator': discriminator.state_dict(),
                            'optimizer_G': optimizer_G.state_dict(),
                            'optimizer_D': optimizer_D.state_dict(),
                        },
                        f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_GAN.pt'
                    )
                pbar.set_postfix(FID=f'{FID:.4f} (best={best_FID:.4f} epoch={best_epoch})')

def train_encoder_izif(good_folds, bad_folds, good_round, bad_round, result_dir):
    mkdir(f'{result_dir}/images_e')

    generator = models.fanogan.Generator().cuda()
    discriminator = models.fanogan.Discriminator().cuda()
    encoder = models.fanogan.Encoder().cuda()
    
    dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)

    model = torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_GAN.pt')
    generator.load_state_dict(model['generator'])
    discriminator.load_state_dict(model['discriminator'])

    generator.eval()
    discriminator.eval()

    criterion = nn.MSELoss()
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=cfg.trainer.lr, betas=(cfg.trainer.b1, cfg.trainer.b2))

    batches_done = 0
    best_auc = 0
    with trange(1, cfg.trainer.encoder_epochs+1, desc=f'{good_round}_{bad_round}') as pbar:
        for epoch in pbar:
            encoder.train()
            for i, (real_imgs, _) in enumerate(dataloaders['train']):
                real_imgs = real_imgs.cuda()
                optimizer_E.zero_grad()

                z = encoder(real_imgs)
                fake_imgs = generator(z)
                real_features = discriminator.forward_features(real_imgs)
                fake_features = discriminator.forward_features(fake_imgs)

                loss_imgs = criterion(fake_imgs, real_imgs)
                loss_features = criterion(fake_features, real_features)
                e_loss = loss_imgs + cfg.trainer.kappa * loss_features
                e_loss.backward()
                optimizer_E.step()

                if i % cfg.trainer.n_critic == 0:
                    log(
                        f"[Epoch {epoch:{len(str(cfg.trainer.num_epochs))}}/{cfg.trainer.num_epochs}] "
                        f"[Batch {i+1:{len(str(len(dataloaders['train'])))}}/{len(dataloaders['train'])}] "
                        f"[E loss: {e_loss.item():3f}]",
                        pt=False,
                    )
                    if batches_done % cfg.trainer.save_image_interval == 0:
                        fake_z = encoder(fake_imgs)
                        reconfiguration_imgs = generator(fake_z)
                        save_image(
                            reconfiguration_imgs.data[:25],
                            f'{result_dir}/images_e/{batches_done:06}.png',
                            nrow=5,
                            normalize=True,
                        )
                    batches_done += cfg.trainer.n_critic
            
            if epoch % cfg.trainer.val_interval == 0:
                validation_auc = validate_fanogan(good_round, bad_round, encoder, dataloaders['validation'], result_dir)
                log(f"Validation AUC: {validation_auc:.4f}", pt=False)
                if validation_auc > best_auc:
                    best_auc = validation_auc
                    best_epoch = epoch
                    torch.save(encoder.state_dict(), f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_encoder.pt')
                    pbar.set_postfix(AUC=f'{validation_auc:.4f} (best={best_auc:.4f} epoch={best_epoch})')

def validate_cae(model, dataloader):
    model.eval()
    criterion = torch.nn.MSELoss()
    losses, labels, anomaly_scores = [], [], []
    with torch.no_grad():
        for image, label in dataloader:
            image = image.cuda()
            _, recon = model(image)
            loss = criterion(recon, image)
            
            losses.append(loss.item())
            labels.append(label.item())
            anomaly_scores.append(loss.item())
    mean_loss = np.mean(losses)
    auc = roc_auc_score(labels, anomaly_scores)
    return mean_loss, auc, labels, anomaly_scores

def validate_pcae(model, dataloader):
    model.eval()
    losses, labels = [], []
    with torch.no_grad():
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
            loss = ((recon - image) ** 2).mean()
            losses.append(loss.item())
            labels.append(label[0, 0].item())
    mean_loss = np.mean(losses)
    anomaly_scores = losses
    auc = roc_auc_score(labels, anomaly_scores)
    return mean_loss, auc, labels, anomaly_scores

def validate_ssae(model, dataloader):
    model.eval()
    criterion = torch.nn.MSELoss()
    labels, anomaly_scores = [], []
    with torch.no_grad():
        for image0, label in dataloader:
            image0 = image0.cuda()
            recon0, recon1, recon2, down_image1, down_image2 = model(image0)
            anomaly_score = criterion(image0, recon0) + criterion(down_image1, recon1) + criterion(down_image2, recon2)
            labels.append(label.item())
            anomaly_scores.append(anomaly_score.item())
    auc = roc_auc_score(labels, anomaly_scores)
    return auc, labels, anomaly_scores

def validate_dagmm(model, dataloaders):
    model.eval()

    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0
    for image, _ in dataloaders['train']:
        image = image.cuda()
        enc, dec, z, gamma = model(image)
        phi, mu, cov = model.compute_gmm_params(z, gamma)
        batch_gamma_sum = gamma.sum(axis=0)
        gamma_sum += batch_gamma_sum
        mu_sum += mu * batch_gamma_sum.unsqueeze(-1)
        cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
        N += image.size(0)
    train_phi = gamma_sum / N
    train_mu = mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    # train_energy = []
    # train_labels = []
    # train_z = []
    # for image, label in dataloaders['train']:
    #     image = image.cuda()
    #     enc, dec, z, gamma = model(image)
    #     sample_energy, cov_diag = model.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
    #     train_energy.append(sample_energy.data.cpu().numpy())
    #     train_z.append(z.data.cpu().numpy())
    #     train_labels.append(label.numpy())
    # train_energy = np.concatenate(train_energy,axis=0)
    # train_z = np.concatenate(train_z,axis=0)
    # train_labels = np.concatenate(train_labels,axis=0)

    # model.phi = train_phi
    # model.mu = train_mu
    # model.cov = train_cov

    labels, anomaly_scores = [], []
    with torch.no_grad():
        for image, label in dataloaders['validation']:
            image = image.cuda()
            enc, dec, z, gamma = model(image)
            sample_energy, cov_diag = model.compute_energy(z, size_average=False)
            labels.append(label.item())
            anomaly_scores.append(sample_energy.item())
            
    auc = roc_auc_score(labels, anomaly_scores)
    return auc

def validate_fanogan(good_round, bad_round, encoder, dataloader, result_dir):
    generator = models.fanogan.Generator().cuda()
    discriminator = models.fanogan.Discriminator().cuda()
    
    model = torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}_GAN.pt')
    generator.load_state_dict(model['generator'])
    discriminator.load_state_dict(model['discriminator'])

    generator.eval()
    discriminator.eval()
    encoder.eval()

    criterion = nn.MSELoss()
    labels, anomaly_scores = [], []
    with torch.no_grad():
        for real_img, label in dataloader:
            real_img = real_img.cuda()

            real_z = encoder(real_img)
            fake_img = generator(real_z)

            real_feature = discriminator.forward_features(real_img)
            fake_feature = discriminator.forward_features(fake_img)

            img_distance = criterion(fake_img, real_img)
            loss_feature = criterion(fake_feature, real_feature)
            anomaly_score = img_distance + cfg.trainer.kappa * loss_feature

            labels.append(label.item())
            anomaly_scores.append(anomaly_score.item())

    auc = roc_auc_score(labels, anomaly_scores)
    return auc

def test_cae(good_folds, bad_folds, result_dir):
    model = models.cae.AutoEncoder().cuda()
    aucs = []
    for good_round in range(cfg.k):
        for bad_round in range(2):
            model.load_state_dict(torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt'))
            model.eval()
            criterion = torch.nn.MSELoss()
            losses, labels, anomaly_scores = [], [], []
            dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)
            with torch.no_grad():
                for image, label in tqdm(dataloaders['test'], desc=f'{good_round}_{bad_round}'):
                    image = image.cuda()
                    _, recon = model(image)
                    loss = criterion(recon, image)

                    losses.append(loss.item())
                    labels.append(label.item())
                    anomaly_scores.append(loss.item())
            mean_loss = np.mean(losses)
            auc = roc_auc_score(labels, anomaly_scores)
            aucs.append(auc)
            with open(f'{result_dir}/test/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
                pickle.dump({'test_loss': mean_loss, 'test_auc': auc, 'test_labels': labels, 'anomaly_scores': anomaly_scores}, f)
    print(f'Test AUC: mean={np.mean(aucs):4f} std={np.std(aucs):4f}')
    return aucs

def test_ssae(good_folds, bad_folds, result_dir):
    model = models.ssae.ScaleSpaceAutoEncoder().cuda()
    aucs = []
    for good_round in range(cfg.k):
        for bad_round in range(2):
            model.load_state_dict(torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt'))
            model.eval()
            criterion = torch.nn.MSELoss()
            labels, anomaly_scores = [], []
            dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)
            with torch.no_grad():
                for image0, label in tqdm(dataloaders['test'], desc=f'{good_round}_{bad_round}'):
                    image0 = image0.cuda()
                    recon0, recon1, recon2, down_image1, down_image2 = model(image0)
                    anomaly_score = criterion(image0, recon0) + criterion(down_image1, recon1) + criterion(down_image2, recon2)
                    labels.append(label.item())
                    anomaly_scores.append(anomaly_score.item())
            auc = roc_auc_score(labels, anomaly_scores)
            aucs.append(auc)
            with open(f'{result_dir}/test/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
                pickle.dump({'test_auc': auc, 'test_labels': labels, 'anomaly_scores': anomaly_scores}, f)
    print(f'Test AUC: mean={np.mean(aucs):4f} std={np.std(aucs):4f}')
    return aucs

def test_dagmm(good_folds, bad_folds, result_dir):
    model = models.dagmm.DAGMM().cuda()
    aucs = []
    for good_round in [0]:# range(5):
        for bad_round in [0]:# range(5):
            dataloaders = build_dataloaders(good_folds, bad_folds, good_round, bad_round)

            model = torch.load(f'{result_dir}/best_models/Good{good_round}_Bad{bad_round}.pt')
            model.eval()

            # N = 0
            # mu_sum = 0
            # cov_sum = 0
            # gamma_sum = 0
            # for image, _ in dataloaders['train']:
            #     image = image.cuda()
            #     enc, dec, z, gamma = model(image)
            #     phi, mu, cov = model.compute_gmm_params(z, gamma)
            #     batch_gamma_sum = gamma.sum(axis=0)
            #     gamma_sum += batch_gamma_sum
            #     mu_sum += mu * batch_gamma_sum.unsqueeze(-1)
            #     cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            #     N += image.size(0)
            # train_phi = gamma_sum / N
            # train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            # train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
            # model.phi = train_phi
            # model.mu = train_mu
            # model.cov = train_cov

            losses, labels, anomaly_scores = [], [], []
            with torch.no_grad():
                for image, label in tqdm(dataloaders['test'], desc=f'{good_round}_{bad_round}'):
                    image = image.cuda()
                    enc, dec, z, gamma = model(image)
                    loss, sample_energy, recon_error, cov_diag = model.loss_function(image, dec, z, gamma, cfg.trainer.lambda_energy, cfg.trainer.lambda_cov_diag)
                    sample_energy, cov_diag = model.compute_energy(z, size_average=False)
                    losses.append(loss.item())
                    labels.append(label.item())
                    anomaly_scores.append(sample_energy.item())
            mean_loss = np.mean(losses)
            auc = roc_auc_score(labels, anomaly_scores)
            aucs.append(auc)
            with open(f'{result_dir}/test/Good{good_round}_Bad{bad_round}.pickle', 'wb') as f:
                pickle.dump({'test_loss': mean_loss, 'test_auc': auc}, f)
    print(f'Test AUC: mean={np.mean(aucs):4f} std={np.std(aucs):4f}')
    return aucs

def test_fanogan(good_folds, bad_folds, good_round, bad_round, result_dir):
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
    labels, anomaly_scores = [], []
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

            labels.append(label.item())
            anomaly_scores.append(anomaly_score.item())

    auc = roc_auc_score(labels, anomaly_scores)
    return auc
