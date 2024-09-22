import json

import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import copy
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from utils.sampling import dataset_iid, dataset_noniid


import torch.nn.functional as F

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_data(config):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform = trans_cifar)
    dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=config["dataset"]["val_split"], random_state=42)

    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)

    if config["training"]["sampling"]== 'iid':
        dict_users = dataset_iid(dataset_train, config["training"]["clients"])
    elif ["training"]["sampling"] == 'noniid':
        dict_users = dataset_noniid(dataset_train, config)
    else:
        raise ValueError('Error: unrecognized sampling')


    return dataset_train, dataset_val,dataset_test, dict_users

class CIFAR_10Dataset(Dataset):
    def __init__(
            self,
            dataset,
            batch_size,
            head_model,
            tail_model,
            idxs,
            gamma
    ):
        self.dataset = dataset
        self.head = head_model
        self.tail = tail_model
        self.batch_size = batch_size
        self.pruned_dataset = self.dataset_pruning(head_model, tail_model, batch_size, dataset, idxs, gamma)
        self.num_example = len(self.pruned_dataset)
        
        self.num_batches = int(
            (self.num_example + self.batch_size - 1) / self.batch_size
        )

    def load_dataset(config):
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dataset.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_train = copy.deepcopy(dataset)
        dataset_test = dataset.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)

        if config["training"]["sampling"] == 'iid':
            dict_users = dataset_iid(dataset_train, config["training"]["clients"])
        elif config["training"]["sampling"] == 'noniid':
            dict_users = dataset_noniid(dataset_train, config)
        else:
            raise ValueError('Error: unrecognized sampling')
        
        return dataset_train, dataset_test, dict_users

    def dataset_pruning(head_model, tail_model, batch_size, dataset, idxs, gamma, config):
        head_model.eval()
        tail_model.eval()

        el2n_scores = []
        dataset_tmp = []

        dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size = batch_size, shuffle = True)

        with torch.no_grad():
            for images, labels in dataloader:
                head_output = head_model(images)

                logits = tail_model(head_output)

                labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()

                softmax_output = F.softmax(logits, dim=1)
                error = softmax_output - labels_one_hot

                el2n_score = torch.norm(error, p=2, dim=1)

                for i in range(len(el2n_score)):
                    el2n_scores.append(el2n_score[i].item())
                    dataset_tmp.append((images[i], labels[i]))


        sorted_index = sorted(range(len(el2n_scores)), key=lambda i: el2n_scores[i], reverse=True)

        n = len(dataset_tmp)
        num_to_keep = int(gamma * n)
        pruned_dataset = [dataset_tmp[i] for i in sorted_index[:num_to_keep]]

        return pruned_dataset
    
    def __len__(self):
        return self.num_batches * self.batch_size
    
    # def __getitem__(self, item):
        # images, labels = self.



def get_dataloader(dataset, head_model, tail_model, idxs, config):
    train_data = CIFAR_10Dataset(
        dataset,
        batch_size=config["dataset"]["train_batch_size"],
        head_model = head_model,
        tail_model = tail_model,
        idxs = idxs,
        gamma = config["training"]["gamma"]
    )
    train_loader = DataLoader(
        train_data,
        batch_size = config["dataset"]["train_batch_size"],
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    num_batches = train_data.num_batches

    return num_batches, train_loader


def dataset_pruning(head_model, tail_model, batch_size, dataset, idxs, gamma, config):
        head_model.eval()
        tail_model.eval()

        el2n_scores = []
        dataset_tmp = []

        dataloader = DataLoader(DatasetSplit(dataset, idxs), batch_size = batch_size, shuffle = True)

        with torch.no_grad():
            for images, labels in dataloader:
                head_output = head_model(images)

                logits = tail_model(head_output)

                labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()

                softmax_output = F.softmax(logits, dim=1)
                error = softmax_output - labels_one_hot

                el2n_score = torch.norm(error, p=2, dim=1)

                for i in range(len(el2n_score)):
                    el2n_scores.append(el2n_score[i].item())
                    dataset_tmp.append((images[i], labels[i]))


        sorted_index = sorted(range(len(el2n_scores)), key=lambda i: el2n_scores[i], reverse=True)

        n = len(dataset_tmp)
        num_to_keep = int(gamma * n)
        pruned_dataset = [dataset_tmp[i] for i in sorted_index[:num_to_keep]]

        return pruned_dataset