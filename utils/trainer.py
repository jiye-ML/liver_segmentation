import torch 
import os 
from tensorboardX import SummaryWriter


class trainer(object):
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 dataloaders, comments, verbose_train, verbose_val,
                 ckpt_frequency,  max_epochs, checkpoint_dir='checkpoints', 
                 start_epoch=0, start_iter=0, device=torch.device('cuda:0')):
        self.model = model 
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_epochs = max_epochs
        self.verbose_train = verbose_train
        self.verbose_val = verbose_val
        self.ckpt_frequency = ckpt_frequency

        self.epoch = 0
        self.iter = start_iter
        self.comments = comments
        self.current_val_loss = 0.0
        self.writer = SummaryWriter(comment=self.comments)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def train(self):
        for self.epoch in range(self.max_epochs):
            current_lr = self.lr_scheduler.get_lr()
            print('Epoch: {}'.format(self.epoch+1))
            print('learning rate: {}'.format(current_lr[-1]))
            self.training_phase(self.dataloaders['train'])
            epoch_val_loss = self.validating_phase(self.dataloaders['val'])
            self.writer.add_scalar('Val/Loss(end of epoch)', epoch_val_loss, self.epoch+1)
            self.writer.add_scalar('Val/Loss', epoch_val_loss, self.iter)
            print('End of the epoch:')
            print('val loss: {:.16f}'.format(epoch_val_loss))
            self.lr_scheduler.step()
        self.writer.close()

    def training_phase(self, dataloader):
        self.model.train()
        for inputs, targets in dataloader:
            self.iter += 1
            outputs = self.model(inputs.to(self.device))
            loss = self.loss_criterion(outputs, targets.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (self.iter % self.verbose_train) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch+1, self.iter))
                print('training loss: {:.16f}'.format(loss.item()))

            if (self.iter % self.verbose_val) == 0:
                self.current_val_loss = self.validating_phase(self.dataloaders['val'])
                self.writer.add_scalar('Val/Loss', self.current_val_loss, self.iter)
                print('Epoch: {}, iter: {}'.format(self.epoch+1, self.iter))
                print('val loss: {:.16f}'.format(self.current_val_loss))
  
            if (self.iter % self.ckpt_frequency) == 0:
                checkpoint_name = os.path.join(self.checkpoint_dir, self.comments + 'iter_' + str(self.iter) + '.pth')
                torch.save(self.model.state_dict(), checkpoint_name)
        
    def validating_phase(self, dataloader):
        self.model.eval()
        loss_total = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs.to(self.device))
                loss = self.loss_criterion(outputs, targets.to(self.device))
                loss_total += loss.item() * inputs.size(0)
        loss_output = loss_total / dataloader.dataset.__len__()
        self.model.train()
        return loss_output