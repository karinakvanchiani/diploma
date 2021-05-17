import copy 
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloaders, criterion, optimizer, num_epochs=25):

    val_acc_history = []

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    model = model.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


def score_model(model, dataloader):
    
    preds, true_labels = [], []

    model.eval() 
    scores = 0
    for data, labels in dataloader:
        outputs = model(data.to(device))
        _, pred = torch.max(outputs, 1)

        true_labels.extend(labels)
        preds.extend(pred)

    for i in range(len(true_labels)):
        true_labels[i] = int(true_labels[i])
        preds[i] = int(preds[i])

    return true_labels, preds