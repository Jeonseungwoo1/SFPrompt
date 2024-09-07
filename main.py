import numpy as np
import argparse
import wandb
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from utils.experiment import load_config
from dataset import load_data, get_dataloader
from network import head_model, body_model, tail_model
from src.update import Local_loss_update, DatasetSplit


def FedAvg(wt):
    wt_avg = copy.deepcopy(wt[0])
    for k in wt_avg.keys():
        for i in range(1, len(wt)):
            wt_avg[k] += wt[i][k]
        wt_avg[k] = torch.div(wt_avg[k], len(wt))
    
    return wt_avg
    
def server_forward_propagation(fx_client, body_model):
    body_model.eval()
    
    server_fx = body_model(fx_client)

    return server_fx

def server_backward_propagation(dfx_client, body_model, server_fx, y):
    body_model.train()

    loss = nn.CrossEntropyLoss(server_fx, y)
    loss.backward(dfx_client)
    dfx_server = server_fx.grad.clone().detach()
    
    return dfx_server

class Client(object):
    def __init__(self, head_model, tail_model, body_model, prompt_module, idx, lr, device, dataset = None, idxs = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.head = head_model
        self.tail = tail_model
        self.body = body_model
        self.prompt = prompt_module
        self. ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = 32, shuffle=True)

    def train(self, head_model, tail_model, body_model, prompt_module):

        #Client forward update
        head_model.eval()
        tail_model.train()
        prompt_module.train()

        optimizer_tail = torch.optim.SGD(tail_model.parameters(), lr = self.lr)
        optimizer_prompt = torch.optim.SGD(prompt_module.parameters(), lr = self.lr)

        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer_tail.zero_grad()
            optimizer_prompt.zero_grad()

            #client forward update
            head_output = self.head(images)
            smashed_data = self.prompt(head_output)

            client_fx = smashed_data.clone().detach().requires_grad_(True)

            #Send smashed data to server and perform server-side forward propagation
            server_fx = server_forward_propagation(client_fx, body_model)

            #Send server-side smashed data to client and perform tail_model forward propagation
            fx_server = server_fx.to(self.device)
            tail_output = self.tail(fx_server)

            loss = nn.CrossEntropyLoss(tail_output, labels)

            #client backward update
            loss.backward()
            dfx_client = client_fx.grad.clone().detach()
            optimizer_tail.step()

            #client update
            dfx_server = server_backward_propagation(dfx_client, body_model, server_fx, labels)
            optimizer_prompt.step()


        return tail_model, prompt_module

            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SFPrompt Script")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if config["wandb"]["logging"]:
        wandb.init(project="SFPrompt-experiments", name=config["wandb"]["run_name"])


    torch.manual_seed(config["distributed"]["random_seed"])    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    dataset_train, dataset_test, dict_users = load_data(config)



    for iter in range(config["training"]["global_epochs"]):
        w_local_tail = []
        local_prompt = []
        
        idxs_users = np.random.choice(range(config["training"]["total_clients"]), config["training"]["clients"], replace=False)
        num_batches, train_dl = get_dataloader(dataset_train, idxs_users, config)

        pruned_dataset = dataset_pruning(head_model, tail_model, dataset_train, config, idxs_users)
        
        for idx in idxs_users:
            w_head, w_tail, p = Local_loss_update(config = config, dataset = train_dl, idxs=set(list(dict_users[idx])))
            w_local_tail[idx] = copy.deepcopy(w_tail)
            local_prompt[idx] = copy.deepcopy(p)

        for idx in idxs_users:
            w_local_tail_, local_prompt_ = Client(head=head_model, tail=w_local_tail[idx], body=body_model, prompt=local_prompt[idx], lr = config["training"]["lr"], device = device, dataset = pruned_dataset, idxs = idxs_users, batch_size = config["dataset"]["train_batch_size"])
            w_local_tail.append(copy.deepcopy(w_local_tail_))
            local_prompt.append(copy.deepcopy(local_prompt_))

        w_glob_tail = FedAvg(w_local_tail)
        global_prompt = FedAvg(local_prompt)
     
