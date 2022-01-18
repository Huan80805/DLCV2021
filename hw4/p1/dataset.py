import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from config import TrainConfig

# fix random seeds for reproducibility
config = TrainConfig()
SEED = config.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# mini-Imagenet dataset
# train set: 38400 84*84 RGB 64 classes
# validation set: 9600 84*84 RGB 16 classes
class MiniTrainset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
                ],p=0.8),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # convert raw_label to num_label
        self.raw_label = self.data_df["label"].tolist() #raw_label
        self.label = []
        self.filename = self.data_df["filename"].tolist()
        raw2num = dict()
        label_set = set(self.raw_label)
        self.label2idx =[] #collect image from same label in term of idx
        for num_label, raw_label in enumerate(label_set):
            raw2num[raw_label] = num_label
            self.label2idx.append([])
        for idx, raw_label in enumerate(self.raw_label):
            self.label.append(raw2num[raw_label])
            self.label2idx[raw2num[raw_label]].append(idx)
        self.label2idx = torch.tensor(self.label2idx, dtype=torch.int)
        #print(raw2num)           
    def __getitem__(self, index):
        path = self.filename[index]
        raw_label = self.raw_label[index]
        label = self.label[index]
        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.label)


class TrainSampler:
    def __init__(self, label2idx, n_batch, n_way, n_shot, n_query):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_batch = n_batch
        self.label2idx = label2idx

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i in range(self.n_batch):
            batch = []
            picked_classes = torch.randperm(len(self.label2idx))[:self.n_way]
            for label in picked_classes:
                idx = self.label2idx[label]
                picked_idx = idx[torch.randperm(len(idx))[:self.n_shot+self.n_query]]
                batch.append(picked_idx)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


# mini-Imagenet Test/Valid dataset
class MiniTestset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]

        image = Image.open(os.path.join(self.data_dir, path))
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data_df)


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence)

    def __len__(self):
        return len(self.sampled_sequence)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #----------------trainset--------------
    # n_way, n_shot, n_query, n_batch = 30, 1, 15, 80
    # trainset = MiniTrainset('../hw4_data/mini/train.csv', '../hw4_data/mini/train')
    # # print(len(trainset.label2idx))      #64
    # # print(len(trainset.label2idx[0]))   #600
    # # print(len(set(trainset.label)))     #64
    # # print(len(set(trainset.raw_label))) #64
    # # print(len(trainset))                #38400
    # # batch_size = n_way*(n_shot+n_query)
    # # n_batch = # of data/batch_size
    # train_sampler = TrainSampler(label2idx=trainset.label2idx,
    #                             n_batch=n_batch,
    #                             n_way=n_way, 
    #                             n_shot=n_shot, 
    #                             n_query=n_query)
    # train_dataloader = DataLoader(dataset=trainset, 
    #                             batch_sampler=train_sampler,
    #                             pin_memory=False, 
    #                             num_workers=4, 
    #                             worker_init_fn=worker_init_fn)
    # train_dataloader = iter(train_dataloader)
    # images, labels = next(train_dataloader)
    # images = images.to(device)
    # # encoding raw_label to relative label
    # _, relative_labels = torch.unique(labels, return_inverse=True)
    # print(labels[:n_way*n_shot], labels[n_way*n_shot:])
    # # print(images.shape)
    # support_labels = relative_labels.clone().detach()[:n_way*n_shot].to(device)
    # query_labels = relative_labels.clone().detach()[n_way*n_shot:].to(device)
    # # print(support_labels)
    # # print(query_labels)

    # --------------testset---------------
    testset = MiniTestset('../hw4_data/mini/val.csv', '../hw4_data/mini/val')
    n_way, n_shot, n_query = 5, 1, 15 #fixed
    test_sampler = GeneratorSampler(episode_file_path='../hw4_data/mini/val_testcase.csv')
    test_dataloader = DataLoader(testset, 
                            batch_size=5*(15 + 1),
                            num_workers=4, 
                            pin_memory=False, 
                            worker_init_fn=worker_init_fn,
                            sampler=test_sampler)
    # print(len(testset))  #9600
    test_dataloader = iter(test_dataloader)
    images, labels = next(test_dataloader)
    images = images.to(device)
    # print(images.shape)
    # print(labels)   #labels: numeric value
    label2relative = {label: relative_label for relative_label, label in enumerate(set(labels))}
    # print(label2relative)
    support_data = images[:n_way*n_shot]
    query_data = images[n_way*n_shot:]
    # print(support_data)
    # print(query_data)
    support_labels = torch.tensor(
        [label2relative[label] for label in labels[:n_way*n_shot]],
        dtype=torch.int
        ).to(device)
    query_labels = torch.tensor(
        [label2relative[label] for label in labels[n_way*n_shot:]],
        dtype=torch.int
        ).to(device)
    # print(support_labels)
    print(query_labels)
    print(label2relative)
    print(labels)

    