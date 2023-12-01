from tqdm import trange
import torch
from config import cfg
from data import *
from model import *
from utils import *
from evaluate import *
from visualization import *
from loss import *


def train():
    '''
    Train and save a model.
    '''

    # data preparation
    mkdir(f'{cfg.PATHS.OUTPUT_DIR}/models')
    dataloaders = build_dataloaders()  # prepare dataloaders
    metrics = Logger(f'{cfg.PATHS.OUTPUT_DIR}/metrics.json')  # create a metric logger
    
    # initialize model, optimizer, and loss function
    model = build_model(cfg.MODEL).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAINER.LR)
    criterion = loss_fn(weight_calculation(dataloaders))
    scaler = torch.cuda.amp.GradScaler()
    
    # model training
    with trange(1, cfg.TRAINER.NUM_EPOCHS+1, desc=f'Train') as pbar:
        for _ in pbar:
            # train mode
            model.train()

            # train model
            losses = []
            for image, label in dataloaders['train']:
                image, label = image.cuda(), label.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = criterion(model(image), label[:, 0])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            metrics.log({
                'train_loss': np.mean(losses)
            })
            
            # evaluate and save model
            for eval_mode in ['validation', 'test']:
                evaluate(model, dataloaders, eval_mode, metrics)
            validation_losses = metrics.get('validation_loss')
            torch.save(model.state_dict(), f'{cfg.PATHS.OUTPUT_DIR}/models/final.pt')
            if validation_losses[-1] == min(validation_losses):
                torch.save(model.state_dict(), f'{cfg.PATHS.OUTPUT_DIR}/models/best.pt')
            pbar.set_postfix(val_loss=f'{validation_losses[-1]:.2f}')

def main():
    assert not os.path.exists(cfg.PATHS.OUTPUT_DIR)
    set_random_seed()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train()
    plots()

if __name__ == '__main__':
    main()
