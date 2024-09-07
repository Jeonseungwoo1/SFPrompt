import json
import os
import torch

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)