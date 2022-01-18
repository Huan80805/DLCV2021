import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import glob
from PIL import Image
from dataset import dataset
from torchvision.transforms import transforms
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
])

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    test_images = glob.glob("output/*.png")
    n = len(test_images)
    corr = 0
    class_corr = [0]*10
    print(n)
    for image in test_images:
        class_label = int(os.path.split(image)[1].split("_")[0])
        image = Image.open(image)
        
        image = transform(image).view(1,3,28,28).to(device)
        pred = net(image)
        pred = torch.argmax(pred, dim=1)
        if pred == class_label:
            corr += 1
            class_corr[class_label] += 1
    print("Acc: {:2f}%".format(100. * corr/n))
    for i in range(10):
        print("Digit {} Acc: {:.2f}".format(i, class_corr[i]))


    
