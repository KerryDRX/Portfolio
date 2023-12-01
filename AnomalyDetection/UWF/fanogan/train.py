from tqdm import trange
import torch
from torchvision.utils import save_image
from config import cfg
from data import build_dataloaders
from model import *
from utils import *
from evaluate import *


def train_gan():
    '''
    Train and save a GAN.
    '''

    # data preparation
    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/generated_images')  # directory to save generated images
    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/model_gan')  # directory to save GAN
    dataloaders = build_dataloaders()  # prepare dataloaders
    train_datalooper = infinitelooper(dataloaders['train'])  # create infinite training data looper
    fid_stats_path = f'{cfg.PATHS.OUTPUT_DIR}/fid_stats.npz'  # statistics of training data (for FID calculation)
    calc_and_save_stats(dataloaders['train_identity3'], fid_stats_path)  # calculate statistics of training images (without augmentation)
    metrics = Logger(f'{cfg.PATHS.OUTPUT_DIR}/metrics_gan.json')  # create a metric logger
    
    # initialize GAN, optimizers, and schedulers
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=cfg.TRAINER.GAN.LR)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.TRAINER.GAN.LR)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lambda x: 1-x/cfg.TRAINER.GAN.NUM_ITERATIONS)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lambda x: 1-x/cfg.TRAINER.GAN.NUM_ITERATIONS)
    
    # model training
    with trange(1, cfg.TRAINER.GAN.NUM_ITERATIONS+1, desc=f'Train') as pbar:
        for iteration in pbar:
            # train mode
            generator.train()
            discriminator.train()
            
            # discriminator training
            losses_D = []
            for _ in range(cfg.TRAINER.GAN.CRITIC):
                real_images = next(train_datalooper)[0].cuda()
                with torch.no_grad():
                    z = torch.randn(cfg.TRAINER.BATCH_SIZE, cfg.DATA.LATENT_DIM).cuda()
                    fake_images = generator(z).detach()
                real_validity = discriminator(real_images)
                fake_validity = discriminator(fake_images)
                gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)
                loss_D = fake_validity.mean() - real_validity.mean() + cfg.TRAINER.GAN.LAMBDA_GP * gradient_penalty
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                losses_D.append(loss_D.item())
            scheduler_D.step()

            # generator training
            enable_grad(discriminator, False)
            z = torch.randn(cfg.TRAINER.BATCH_SIZE, cfg.DATA.LATENT_DIM).cuda()
            fake_images = generator(z)
            fake_validity = discriminator(fake_images)
            loss_G = -fake_validity.mean()
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            enable_grad(discriminator, True)
            scheduler_G.step()
            
            # save and print losses
            metrics.log({
                'loss_D': np.mean(losses_D),
                'loss_G': loss_G.item(),
            })
            pbar.set_postfix({
                'loss_D': f'{np.mean(losses_D):.1f}',
                'loss_G': f'{loss_G.item():.1f}',
            })

            # model evaluation
            if iteration % cfg.TRAINER.GAN.EVAL_INTERVAL == 0:
                # evaluation mode
                generator.eval()
                discriminator.eval()

                # save 25 generated images
                fake_images = fake_images.detach().cpu()[:25]
                fake_images = (fake_images + 1) / 2
                fake_images = torch.concat([fake_images, torch.zeros_like(fake_images[:, :1])], dim=1)
                save_image(
                    fake_images,
                    f'{cfg.PATHS.OUTPUT_DIR}/generated_images/{iteration:05}.jpg',
                    nrow=5,
                    normalize=True,
                )
            
                # evaluate the model
                IS, FID = evaluate_generator(generator, fid_stats_path)
                metrics.log({
                    'IS': IS,
                    'FID': FID,
                })
                pbar.write(f'Iteration {iteration}/{cfg.TRAINER.GAN.NUM_ITERATIONS}: IS={IS[0]:.3f}±{IS[1]:.3f} FID={FID:.1f} {cuda_memory()}')
                
                # save the model
                checkpoint = {
                    'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(), 'optimizer_D': optimizer_D.state_dict(),
                    'scheduler_G': scheduler_G.state_dict(), 'scheduler_D': scheduler_D.state_dict(),
                }
                torch.save(checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/model_gan/final.pt')
                if FID == min(metrics.get('FID')):
                    torch.save(checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/model_gan/best.pt')

def train_encoder():
    '''
    Train and save an encoder.
    '''

    # data preparation
    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/model_encoder')  # directory to save encoder
    dataloaders = build_dataloaders()  # prepare dataloaders
    metrics = Logger(f'{cfg.PATHS.OUTPUT_DIR}/metrics_encoder.json')  # create a metric logger

    # load trained GAN
    checkpoint = torch.load(f'{cfg.PATHS.OUTPUT_DIR}/model_gan/final.pt')
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator.eval()
    discriminator.eval()
    enable_grad(generator, False)
    enable_grad(discriminator, False)

    # initialize encoder and optimizer
    encoder = Encoder().cuda()    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.TRAINER.ENCODER.LR, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    
    # model training
    with trange(1, cfg.TRAINER.ENCODER.NUM_EPOCHS+1, desc=f'Train') as pbar:
        for epoch in pbar:
            # train mode
            encoder.train()

            # train encoder
            losses_recon, losses_feat = [], []
            for real_images, _ in dataloaders['train']:
                real_images = real_images.cuda()
                with torch.cuda.amp.autocast():
                    fake_images = generator(encoder(real_images))
                    real_features = discriminator.extract_features(real_images)
                    fake_features = discriminator.extract_features(fake_images)
                    loss_recon = criterion(fake_images, real_images)
                    loss_feat = criterion(fake_features, real_features)
                    loss = loss_recon + cfg.TRAINER.ENCODER.KAPPA * loss_feat
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses_recon.append(loss_recon.item())
                losses_feat.append(loss_feat.item())
            
            # save losses
            metrics.log({
                'loss_recon': np.mean(losses_recon),
                'loss_feat': np.mean(losses_feat),
            })

            # model evaluation
            if epoch % cfg.TRAINER.ENCODER.EVAL_INTERVAL == 0:
                # evaluation mode
                encoder.eval()

                # calculate, save, and print validation AUC
                validation_auc = evaluate_encoder(generator, discriminator, encoder, dataloaders['validation'])[0]
                metrics.log({
                    'validation_auc': validation_auc,
                })
                pbar.set_postfix(AUC=f'{validation_auc:.3f}')

                # save the model
                checkpoint = {
                    'encoder': encoder.state_dict(),
                    'optimizer_E': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/model_encoder/final.pt')
                if validation_auc == max(metrics.get('validation_auc')):
                    torch.save(checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/model_encoder/best.pt')

def main():
    assert not os.path.exists(cfg.PATHS.OUTPUT_DIR)
    set_random_seed()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train_gan()
    train_encoder()

if __name__ == '__main__':
    main()
