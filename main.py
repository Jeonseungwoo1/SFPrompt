import numpy as np
import argparse
import wandb
import copy
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from utils.experiment import AverageMeter, load_config
from utils.optimizer import get_optimizer
from dataset import load_data, get_dataloader, dataset_pruning
from network import head_model, body_model, tail_model, Head, Body, Tail
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


## SplitLoRA 처럼 구현 바꾸기

"""class Client(object):
    def __init__(self, head_model, tail_model, body_model, prompt_module, idx, lr, device, dataset = None, idxs = None, config = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.head = head_model
        self.tail = tail_model
        self.body = body_model
        self.prompt = prompt_module
        self.config = config
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.config["dataset"]['train_batch_size'], shuffle=True)
        self.pruned_dataset = self.dataset_pruning()

    def dataset_pruning(self):
        self.head.eval()
        self.tail.eval()

        el2n_scores = []
        dataset_tmp = []


        with torch.no_grad():
            for images, labels in self.ldr_train:
                head_output = self.head(images)

                logits = self.tail(head_output)

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
    
    def train(self):

        #Client forward update
        head_model = copy.deepcopy(self.head)
        tail_model = copy.deepcopy(self.tail)
        prompt_module = copy.deepcopy(self.prompt)

        head_model.eval()
        tail_model.train()
        prompt_module.train()

        #optimizer 부분 수정 필요
        optimizer_tail = torch.optim.SGD(tail_model.parameters(), lr = self.lr)


        for batch_idx, (images, labels) in enumerate(self.pruned_dataset):
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


        return tail_model, prompt_module"""

            

def optimizer_step(
    loss,
    optimizer1, head_model, hidden_states1,
    optimizer2, body_model, hidden_states2,
    optimizer3, tail_model,
    schedules=None,
    clip_value=None,
    is_update=True
):
    # Backward pass through all networks
    loss.backward()

    # Store gradients for hidden states
    grad1 = hidden_states1.grad.clone().detach()
    grad2 = hidden_states2.grad.clone().detach()

    # Update model3 (last in the sequence)
    if is_update and clip_value is not None:
        torch.nn.utils.clip_grad_norm_(tail_model.parameters(), clip_value)
    optimizer3.step()
    optimizer3.zero_grad()

    # Update model2
    hidden_states2.backward(grad1)
    if is_update and clip_value is not None:
        torch.nn.utils.clip_grad_norm_(body_model.parameters(), clip_value)
    optimizer2.zero_grad()

    # Update model1 (first in the sequence)
    hidden_states1.backward(grad2)
    if is_update and clip_value is not None:
        torch.nn.utils.clip_grad_norm_(head_model.parameters(), clip_value)
    optimizer1.step()
    optimizer1.zero_grad()



def evaluate(device, head, body, tail, idxs, dataset_val):
    device= device
    head.eval()
    body.eval()
    tail.eval()


def train(
    config,
    device,
    head,
    body,
    tail,
    dataset,
    idxs_users,
    idxs
):
    dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size=config["dataset"]["train_batch_size"], shuffle = True)

    pruned_dataset = dataset_pruning(
        head_model=head,
        tail_model=tail,
        batch_size = config["dataset"["train_batch_size"]],
        dataset = dataset,
        idxs = idxs,
        gamma = config["training"]["gamma"]
    )

    head.train()
    body.train()
    tail.train()

    print("Training Start", config["training"]["epochs"])
    

    device = device

    images = []
    labels = []

    for batch_idx, (image, label) in pruned_dataset:
        images.append(image)
        labels.append(label)
    for idx, (image, label) in pruned_dataset:
        images = images.to(device)
        head_output = head(images)

        head_smashed_data = head_output.clone().detach().requires_grad_(True)

        body_output = body(head_smashed_data)
        body_smashed_data = body_output.clone().detach().requires_grad_(True)

        tail_output = tail(body_smashed_data)
        loss = nn.CrossEntropyLoss(tail_output, labels)

        head.freeze_non_prompt_parameters()
        head_optimizer = optim.Adam([head.prompt_embeddings], lr=config["training"]["lr"])
        head_optimizer = get_optimizer(head, config)
        body_optimizer = get_optimizer(body, config)
        tail_optimizer = get_optimizer(tail, config)
    
        optimizer_step(
            loss,
            head_optimizer, head, head_smashed_data,
            body_optimizer, body, body_smashed_data,
            tail_optimizer, tail
        )

    
    return head.state_dict(), tail.state_dict(), loss


    

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

    dataset_train, dataset_val, dataset_test, dict_users = load_data(config)
    head = copy.deepcopy(Head)
    body = copy.deepcopy(Body)
    tail = copy.deepcopy(Tail)

    avg_loss = AverageMeter()
    
    log_start_time = time.time()
    #global round
    for iter in range(config["training"]["global_epochs"]):

        if iter != 0:
            head = global_prompt
            tail = w_glob_tail

        w_local_tail = []
        w_local_head = []
        
        idxs_users = np.random.choice(range(config["training"]["total_clients"]), config["training"]["clients"], replace=False)
        
        #Local-loss update
        for idx in idxs_users:
            w_head, w_tail= Local_loss_update(head, tail, config = config, dataset = dataset_train, idxs=set(list(dict_users[idx])))
            w_local_tail[idx] = copy.deepcopy(w_tail)
            w_local_head[idx] = copy.deepcopy(w_head)

        #Split training
        for idx in idxs_users:
            w_local_tail_, w_local_head_, loss= train(
                                            config=config,
                                            device=device,
                                            head=w_local_head[idx],
                                            body=body_model,
                                            tail=w_local_tail[idx],
                                            dataset = dataset_train,
                                            idxs_users=idxs_users,
                                            idxs=set(list(dict_users[idx]))
                                            )
            w_local_tail.append(copy.deepcopy(w_local_tail_))
            w_local_head.append(copy.deepcopy(w_local_head_))
            avg_loss.update(loss.item())


        w_glob_tail = FedAvg(w_local_tail)
        global_prompt = FedAvg(w_local_head)
        elapsed = time.time() - log_start_time
        log_str = {
            f"| epoch {iter:3d} | avg loss {avg_loss.avg:5.2f} |"
        }
        print(log_str)
        if config["wandb"]["logging"]:
            wandb.log(
                {
                    "train/epoch_loss": avg_loss.avg,
                    "train/epoch": iter,
                    "train/elapsed_time": elapsed
                }
            )
        log_start_time = time.time()
        avg_loss.reset()


     
