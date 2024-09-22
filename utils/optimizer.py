import torch
import torch.optim as optim

def get_optimizer(model, config):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        eps=config["optimizer"]["adam_epsilon"]
    )

    return optimizer