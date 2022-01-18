import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from vgg16_mod import vgg16_mod
from dataset import dataset

trainset = dataset(root='../hw1_data/p1_data/train_50',transform_aug=True)
# load the testset
valset = dataset(root='../hw1_data/p1_data/val_50',transform_aug=False)

print('# images in trainset:', len(trainset)) # 22500
print('# images in valset:', len(valset)) # 2500

# Use the torch dataloader to iterate through the dataset

trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True)
valset_loader = DataLoader(valset, batch_size=64, shuffle=False)


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model = vgg16_mod()
model.to(device)
def train(model, trainset_loader, testset_loader, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
            save_checkpoint('%i.pth' %ep, model, optimizer)

    
    # save the final model
    save_checkpoint('final.pth',model, optimizer)

def test(model, testset_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

train(model, trainset_loader, valset_loader, epoch=20, log_interval=5)
