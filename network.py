import torch
import torch.nn as nn


from torch.nn.modules.utils import _pair
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from os.path import join as pjoin
import copy
import math


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        #self.hybrid = None

        patch_size = (16, 16)  # Set patch size to 4x4
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config["prompt"]["hidden_size"],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config["prompt"]["hidden_size"]))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["prompt"]["hidden_size"]))

        self.dropout = nn.Dropout(config["prompt"]["dropout_rate"])


    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Head(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Head, self).__init__()

        # Indent consistently with 4 spaces
        self.embeddings = Embeddings(config, img_size, in_channels)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["prompt"]["hidden_size"]))
        self.dropout = nn.Dropout(config["prompt"]["dropout_rate"])

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, config["prompt"]["num_token"], config["prompt"]["hidden_size"]))
        nn.init.uniform_(self.prompt_embeddings.data, -0.1, 0.1)
    
    def forward(self, x):
        B = x.size(0)

        # Get the patch and position embeddings from the Embeddings class
        patch_embeddings = self.embeddings(x)

        # Expand prompt tokens to match the batch size
        prompt_tokens = self.prompt_embeddings.expand(B, -1, -1)

        # Concatenate cls_token, prompt_tokens, and patch_embeddings
        x = torch.cat((self.cls_token.expand(B, -1, -1), prompt_tokens, patch_embeddings), dim=1)

        x = self.dropout(x)

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
