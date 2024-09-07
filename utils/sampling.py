import numpy as np
from torchvision import datasets, transforms

def dataset_iid(dataset, num_users = None):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def dataset_noniid(dataset, config):
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(config["training"]["clients"])}

    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num+1, size=config["training"]["clients"])
    print(f"Total number of datasets owned by client: {sum(random_num_size)}")

    assert num_dataset >= sum(random_num_size)

    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set
        
    return dict_users