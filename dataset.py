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
        dict_users = dataset_iid(dataset_train, config["training"]["total_clients"])
    elif ["training"]["sampling"] == 'noniid':
        dict_users = dataset_noniid(dataset_train, config)
    else:
        raise ValueError('Error: unrecognized sampling')


    return dataset_train, dataset_val,dataset_test, dict_users


def dataset_pruning(head_model, tail_model, batch_size, dataset, idxs, gamma, device, config):
        head_model = copy.deepcopy(head_model)
        tail_model = copy.deepcopy(tail_model)
        head_model.eval()
        tail_model.eval()

        el2n_scores = []
        dataset_tmp = []

        dataset_split = DatasetSplit(dataset, idxs)
        dataloader = DataLoader(dataset_split, batch_size = batch_size, shuffle = True)

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

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