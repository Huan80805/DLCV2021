from pickle import MEMOIZE
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from model import VGG16_FCN32, VGG16_FCN8
from dataset import dataset
import numpy as np

def train(model, trainset_loader, testset_loader, epoch, log_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
              pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  ep, batch_idx * len(data), len(trainset_loader.dataset),
                  100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        test(model, testset_loader)
        if ep % 5 == 0 :
            save_checkpoint('FCN32_%i.pth' %ep, model, optimizer)
    # save the final model
    save_checkpoint('FCN32_final.pth',model, optimizer)

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        #print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def test(model, testset_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    MIOU = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1].reshape((-1,target.shape[1],target.shape[2])) # get the index of the max log-probability
            MIOU += mean_iou_score(pred, target)
    test_loss /= len(testset_loader)
    MIOU /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, MIOU: {:.4f}\n'.format(test_loss, MIOU))


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)



if __name__ == "__main__":
    trainset = dataset(root='../hw1_data/p2_data/train',flip=True)
    # load the testset
    valset = dataset(root='../hw1_data/p2_data/validation',flip=False)

    print('# images in trainset:', len(trainset)) 
    print('# images in testset:', len(valset)) 

    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    valset_loader = DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    model = VGG16_FCN32()
    model.to(device)
    train(model, trainset_loader, valset_loader, epoch=40, log_interval=5)
    
    #test(model, valset_loader)




