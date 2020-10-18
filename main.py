import argparse
from utils.trainer import trainer
from utils.dataset import LITS_dataset, make_dataloaders
import torch 
import torch.nn as nn 
from model.unet import UNet


if __name__ == '__main__':
    device = torch.device('cuda:0')

    LEARNING_RATE = 1e-3
    LR_DECAY_STEP = 2
    LR_DECAY_FACTOR = 0.5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 4
    MAX_EPOCHS = 30
    MODEL = UNet(1, 2).to(device)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    CRITERION = nn.CrossEntropyLoss().to(device)

    tr_path_raw = 'data/tr/raw'
    tr_path_label = 'data/tr/label'
    ts_path_raw = 'data/ts/raw'
    ts_path_label = 'data/ts/label'

    checkpoints_dir = 'checkpoints'
    checkpoint_frequency = 1000
    dataloaders = make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, BATCH_SIZE, n_workers=4)
    comment = 'liver_segmentation_U-Net_on_LITS_dataset_'
    verbose_train = 1
    verbose_val = 500

    trainer = trainer(MODEL, OPTIMIZER, LR_SCHEDULER, CRITERION, dataloaders, comment, verbose_train, verbose_val, checkpoint_frequency, MAX_EPOCHS, checkpoint_dir=checkpoints_dir, device=device)
    trainer.train()



