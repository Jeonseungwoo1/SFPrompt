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
from dataset import load_data, dataset_pruning
from network import Head, Body, Tail
from src.update import Local_loss_update, DatasetSplit


def FedAvg(wt):
    wt_avg = copy.deepcopy(wt[0])
    for k in wt_avg.keys():
        for i in range(1, len(wt)):
            wt_avg[k] += wt[i][k]
        wt_avg[k] = torch.div(wt_avg[k], len(wt))
    
    return wt_avg
            

def optimizer_step(
    loss,
    head_optimizer, head_model, head_smashed_data,
    body_optimizer, body_model, body_smashed_data,
    tail_optimizer, tail_model, tail_grad,
    config,
    is_update=True
):
    # Backward pass through all networks
    loss.backward(retain_graph=True)
    tail_grad.retain_grad()
    body_smashed_data.retain_grad()
    # Store gradients for hidden states
    grad1 = tail_grad.grad
    grad2 = body_smashed_data.grad

    # Update model_tail (last in the sequence)
    if is_update and config["training"]["clip"] > 0:
        torch.nn.utils.clip_grad_norm_(tail_model.parameters(), config["training"]["clip"])
    tail_optimizer.step()
    tail_optimizer.zero_grad()

    # Update model_body
    body_smashed_data.mean().backward(grad1, retain_graph=True)
    if is_update and config["training"]["clip"] > 0:
        torch.nn.utils.clip_grad_norm_(body_model.parameters(), config["training"]["clip"])
    body_optimizer.zero_grad()

    # Update model_head (first in the sequence)
    if is_update and config["training"]["clip"] > 0:
        torch.nn.utils.clip_grad_norm_(head_model.parameters(), config["training"]["clip"])
    head_optimizer.step()
    head_optimizer.zero_grad()




def evaluate(config, device, head, body, tail, idxs, dataset_test):
    device= device
    head.eval()
    body.eval()
    tail.eval()

    correct = 0
    total = 0 
    loss_metter = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            images = images.to(device)
            labels = labels.to(device)

            head_output = head_model(images)
            head_smashed_data = head_output.detach()

            body_output = body_model(head_smashed_data)
            body_smashed_data = body_output.detach()

            outputs = tail_model(body_smashed_data)

            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = loss_meter.avg

    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if config["wandb"]["logging"]:
        wandb.log({
            "test/accuracy": accuracy,
            "test/loss": avg_loss
        })

    return avg_loss, accuracy


def train(
    config,
    device,
    head,
    body,
    tail,
    head_weight,
    tail_weight,
    dataset,
    idxs_users,
    idxs
):


    head.load_state_dict(head_weight)
    tail.load_state_dict(tail_weight)

    head = head.to(device)
    body = body.to(device)
    tail = tail.to(device)

    pruned_dataset = dataset_pruning(
        head_model=head,
        tail_model=tail,
        batch_size = config["dataset"]["train_batch_size"],
        dataset = dataset,
        idxs = idxs,
        gamma = config["training"]["gamma"],
        device=device,
        config = config
    )

    head.train()
    body.train()
    tail.train()

    print("Training Start")
    

    device = device
    
    for idx, (images, labels) in enumerate(pruned_dataset):
        images = images.unsqueeze(0).to(device)
        if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
        labels = labels.to(device)

        head_output = head(images)
      
        head_smashed_data = head_output.clone().detach().requires_grad_(True)

        body_output = body(head_output)
        body_smashed_data = body_output.clone().detach().requires_grad_(True)


        tail_output = tail(body_output)
        tail_grad = tail_output.clone().detach().requires_grad_(True)

        loss = nn.CrossEntropyLoss()(tail_output, labels)

        head.freeze_non_prompt_parameters()
        head_optimizer = optim.Adam([head.prompt_embeddings], lr=config["training"]["lr"])
        body_optimizer = get_optimizer(body, config)
        tail_optimizer = get_optimizer(tail, config)
    
        optimizer_step(
            loss,
            head_optimizer, head, head_smashed_data,
            body_optimizer, body, body_smashed_data,
            tail_optimizer, tail, tail_grad,

            config
        )

    
    return head.state_dict(), tail.state_dict(), loss

def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    

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
    head = Head(config, img_size=(32, 32), in_channels=3)
    body = Body(config)
    tail = Tail(config, config["prompt"]["num_classes"])

    avg_loss = AverageMeter()
    
    log_start_time = time.time()
    #global round
    
    for iter in range(config["training"]["global_epochs"]):
        print(f"Start {iter:3d} Round")
        if iter != 0:
            load_partial_state_dict(head, global_prompt)
            load_partial_state_dict(tail, w_glob_tail)

        w_local_tail = {}
        w_local_head = {}
        
        #idxs_users = np.random.choice(range(config["training"]["total_clients"]), config["training"]["clients"], replace=False)
        idxs_users = np.random.choice(list(dict_users.keys()), config["training"]["clients"], replace=False)

        
        #Local-loss update
        print("Start Local Loss Update")
        for idx in idxs_users:
            loss_update= Local_loss_update(head, tail, config = config, device = device, dataset = dataset_train, idxs=set(list(dict_users[idx])))
            w_head, w_tail= loss_update.update()
            w_local_tail[idx] = copy.deepcopy(w_tail)
            w_local_head[idx] = copy.deepcopy(w_head)

        
        #Split training
        print("Start Split Training")
        for idx in idxs_users:
            w_local_tail_, w_local_head_, loss= train(
                                            config=config,
                                            device=device,
                                            head = head,
                                            body=body,
                                            tail=tail,
                                            head_weight=w_local_head[idx],
                                            tail_weight=w_local_tail[idx],
                                            dataset = dataset_train,
                                            idxs_users=idxs_users,
                                            idxs=set(list(dict_users[idx]))
                                            )
            w_local_tail[idx] = copy.deepcopy(w_local_tail_)
            w_local_head[idx] = copy.deepcopy(w_local_head_)
            avg_loss.update(loss.item())


        w_glob_tail = FedAvg(list(w_local_tail.values()))
        global_prompt = FedAvg(list(w_local_head.values()))
        elapsed = time.time() - log_start_time
        log_str = {
            f"| epoch {iter:3d} | loss {avg_loss.val:5.2f} |"
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

    test_loader = DataLoader(dataset_test, batch_size=config["dataset"]["test_batch_size"], shuffle=False)
    test_loss, test_accuracy = evaluate(config, device, head, body, tail, test_loader)


     
