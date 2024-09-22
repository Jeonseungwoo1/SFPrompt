import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
import numpy as np
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

class Local_loss_update(object):
    def __init__(self, head_model, tail_model, config, device, dataset=None, idxs=None):
        self.config = config
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size= self.config["dataset"]["train_batch_size"], shuffle = True)
        self.head = head_model
        self.tail = tail_model
        self.pruned_dataset = self.dataset_pruning()

    def dataset_pruning(self):
        head_model = copy.deepcopy(self.head)
        tail_model = copy.deepcopy(self.tail)

        head_model = head_model.to(self.device)
        tail_model = tail_model.to(self.device)

        head_model.eval()
        tail_model.eval()

        el2n_scores = []
        dataset_tmp = []


        with torch.no_grad():
            for images, labels in self.dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)

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
        num_to_keep = int(self.config["training"]["gamma"] * n)
        pruned_dataset = [dataset_tmp[i] for i in sorted_index[:num_to_keep]]

        return pruned_dataset


    def update(self):
        head_model = copy.deepcopy(self.head)
        tail_model = copy.deepcopy(self.tail)


        head_model.train()
        tail_model.train()

        head_model = head_model.to(self.device)
        tail_model = tail_model.to(self.device)
        
        optimizer_head = torch.optim.Adam([head_model.prompt_embeddings], lr = self.config["training"]["lr"])
        optimizer_tail = torch.optim.Adam(tail_model.parameters(), lr=self.config["training"]["lr"])

        criterion = self.loss_func

        pruned_dataset = self.pruned_dataset


        for epoch in range(self.config["training"]["local_epochs"]):
            #print(f"Epoch {epoch+1}/{self.config['training']['local_epochs']}")
            for batch_idx, (images,labels) in enumerate(pruned_dataset):
                images = images.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                head_output = head_model(images)
                logits = tail_model(head_output)

                loss = criterion(logits, labels)

                optimizer_head.zero_grad()
                optimizer_tail.zero_grad()
                
                loss.backward()
                
                optimizer_tail.step()
                optimizer_head.step()
                
    
        return head_model.state_dict(), tail_model.state_dict()