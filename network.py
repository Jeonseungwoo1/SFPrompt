import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from torch.nn.modules.utils import _pair
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from os.path import join as pjoin
import copy
import math

"""class head_model(nn.Module):
    def __init__(self, vit_model, num_layers):
        super(head_model, self).__init__()
        self.patch_embdding = vit_model.embeddings
        self.transformer_layers = nn.ModuleList(vit_model.encoder.layer[:num_layers])


    def forward(self, x):
        x = self.patch_embdding(x)
        for layer in self.transformer_layers:
            x = layer(x)

        return x
    

class body_model(nn.Module):
    def __init__(self, vit_model, start_layer, end_layer):
        super(body_model, self).__init__()
        self.transformer_layers = nn.ModuleList(vit_model.encoder.layer[start_layer:end_layer])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)
        return x


class tail_model(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(tail_model, self).__init__()

        self.transformer_layers = nn.ModuleList(vit_model.encoder.layer[-1:])

    def forward(self, x):
        for layer in self.transformer_layers:
            x = layer(x)

        x = self.layernorm(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x    


#이게 맞는지 모르겠음
class prompt_module(nn.Module):
    def __init__(self, prompt_length,  embedding_dim):
        super(prompt_module, self).__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim))


    def forward(self, x):
        batch_size = x.size(0)
        prompt = self.prompt_unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat((prompt,x), dim=1)"""

class Head(nn.Module):
    def __init__(self, config):
        super(Head, self).__init__()
        self.embeddings = nn.Linear(config["prompt"]["hidden_size"], config["prompt"]["hidden_size"])
        self.cls_token = nn.Parameter(torch.zeros(1,1, config["prompt"]["hidden_size"]))
        self.dropout = nn.Dropout(config["prompt"]["dropout_rate"])

        self.config = config
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, config["prompt"]["num_token"], config["prompt"]["hidden_size"]))
        nn.init.uniform_(self.prompt_embeddings.data, -0.1, 0.1)
    
    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expend(B, -1, -1)

        prompt_tokens = self.prompt_embeddings.expand(B, -1, -1)

        x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        x = self.dropout(self.embeddings(x))

        return x
    
    def freeze_non_prompt_parameters(self):
        for name, param in self.named_parameters():
            if name != "prompt_embeddings":
                param.requires_grad=False
    
class Body(nn.Module):
    def __init__(self, config):
        super(Body, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(config["prompt"]["hidden_size"], config["prompt"]["num_heads"])
             for _ in range(config["prompt"]["num_layers"])]
        )
        
        
    def forward(self, x):
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
        return x
    
class Tail(nn.Module):
    def __init__(self, config, num_classes):
        super(Tail, self).__init__()
        self.norm = nn.LayerNorm(config["prompt"]["hidden_size"], eps=1e-6)
        self.head = nn.Linear(config["prompt"]["hidden_size"], num_classes)

    def forward(self, x):
        x = self.norm(x[:,0])
        logits = self.head(x)
        return logits
