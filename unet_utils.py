import time
import copy
import numpy as np
import torch
import torch.nn as nn
from matplotlib.pyplot import plt
from IPython.display import clear_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(device, model, data_tr, data_val, dataset_sizes,
          criterion, optimizer, scheduler=None,
          metric=DiceScore(), num_epochs=25):
  
    since = time.time()
    
    X_val, Y_val = next(iter(data_val))

    history_loss = {'train': [], 'val': []}
    history_metric = {'train': [], 'val': []}

    best_model = copy.deepcopy(model.state_dict())
    best_metric = - np.inf

    for epoch in range(num_epochs):

        model.train()
        phase = 'train'

        running_loss = 0.0
        running_metric = 0.0

        for data, labels in data_tr:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                outputs = model(data)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_metric += metric(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()) * data.size(0)

        if scheduler: scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_metric = running_metric / dataset_sizes[phase]
            
        history_loss[phase].append(epoch_loss)
        print('epoch {}/{} || {} loss: {:.4f} || {}: {:.4f}'.format(
              epoch, num_epochs - 1, phase, epoch_loss, metric.name, epoch_metric))
        history_metric[phase].append(epoch_metric)


        model.eval()
        phase = 'val'

        running_loss = 0.0
        running_metric = 0.0

        Y_hat = model(X_val.to(device)).cpu().detach().numpy() > 0.5

        for data, labels in data_val:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=='train'):
                outputs = model(data)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * data.size(0)
            running_metric += metric(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()) * data.size(0)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_metric = running_metric / dataset_sizes[phase]
            
        history_loss[phase].append(epoch_loss)
        print('epoch {}/{} || {} loss: {:.4f} || {}: {:.4f}'.format(
              epoch, num_epochs - 1, phase, epoch_loss, metric.name, epoch_metric))
        history_metric[phase].append(epoch_metric)

        if phase == 'val' and (epoch_metric > best_metric):
            best_metric = epoch_metric
            best_model = copy.deepcopy(model.state_dict())
      
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k + 1)
            plt.imshow(np.rollaxis(Y_val[k].numpy(), 0, 3)[:, :, 0], cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k + 7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('epoch {}/{} || {} loss: {:.4f} || {}: {:.4f}'.format(
                     epoch + 1, num_epochs, phase, epoch_loss, metric.name, epoch_metric))
        plt.show()


    time_elapsed = time.time() - since
    
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('best val {}: {:4f}'.format(metric.name, best_metric))

    model.load_state_dict(best_model)
    
    return {'model': model, 'history_loss': history_loss, 'history_metric': history_metric}


def score_model(model, metric, dataloader, dataset_size, mode):
    
    model.eval() 
    scores = 0
    for data, labels in dataloader:
        outputs = model(data.to(device)) > 0.5
        scores += metric(labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()) * data.size(0)

    print(f'score on {mode} images = {round(scores/dataset_size, 3)}')


def dice_score(y_true, y_pred, smoothing=1e-6):

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()

    return (2. * intersection + smoothing) / (union + smoothing)


class DiceScore():
    def __init__(self, threshold=0.5, smoothing=1e-6):
        self.name = 'DSC'
        self.smoothing = smoothing
        self.target = 'max'
        self.threshold = 0.5
        
    def __call__(self, y_true, y_pred):
        
        y_pred[y_pred >= self.threshold] = 1.
        y_pred[y_pred <= self.threshold] = 0.
        
        dscs = np.array(list(map(dice_score, y_true, y_pred, [self.smoothing for _ in range(y_pred.shape[0])])))
        
        return np.mean(dscs)


class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        
        y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], -1)
        
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1. - dsc