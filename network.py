import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class head_model(nn.Module):
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
        return torch.cat((prompt,x), dim=1)
