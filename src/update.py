import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

import numpy as np

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
    def __init__(self, config, dataset=None, idxs=None):
        self.config = config
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
    def update(self, head, tail, prompt):

        head.eval()
        tail.train()
        prompt.train()

        optimizer_tail = torch.optim.SGD(tail.parameters(), lr=self.config["training"["lr"]])
        optimizer_prompt = torch.optim.SGD(prompt.parameters(), lr=self.config["training"["lr"]])

        for epoch in range(self.config["training"]["local_epochs"]):
            for batch_idx, (images,labels) in enumerate(self.dataset):
                optimizer_tail.zero_grad()
                optimizer_prompt.zero_grad()

                with torch.no_grad():
                    head_output = head(images)

                enhanced_embedding = prompt(head_output)
                logits = tail(enhanced_embedding)

                loss = self.loss_func(logits, labels)

                loss.backward()

                optimizer_tail.step()
                optimizer_prompt.step()
    
        return head, tail, prompt
    