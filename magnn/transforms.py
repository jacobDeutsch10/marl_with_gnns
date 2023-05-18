import torch
from torch_geometric.data import Data, Batch
import pdb
import numpy as np
def get_adjacent(index, shape, hops=1):
    adjacent = []
    for c in range(shape[2]):
        for dx in range(-hops, hops+1):
            for dy in range(-hops, hops+1):
                if dx != 0 or dy != 0:  # exclude the cell itself
                    adjacent.append((index[0] + dx, index[1] + dy, c))
    valid_adjacent = [i for i in adjacent if 0 <= i[0] < shape[0] and 0 <= i[1] < shape[1]]
    return valid_adjacent

def obs_to_graph(observation, hops=1):
    nodes = []
    edge_index = []
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            for k in range(observation.shape[2]):
                if observation[i, j, k] > 0:
                    nodes.append(((i, j, k), k))
    node_dict = {node[0]: idx for idx, node in enumerate(nodes)}
    for node in nodes:
        adjacents = get_adjacent(node[0], observation.shape, hops=hops)
        for adj in adjacents:
            if adj in node_dict:
                edge_index.append(np.array((node_dict[node[0]], node_dict[adj])))
    node_features = [np.array(node[1]) for node in nodes]
    """if len(node_dict) == 0:
        # do this so we dont error on dummy batch
        []
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features = torch.tensor([node[0] for node in nodes], dtype=torch.float)"""
    return edge_index, node_features

def obs_to_graph_batch(obs):
    hops = 1
    datas = []
    for batch_id, observation in enumerate(obs):
        data = obs_to_graph(observation, hops=hops)
        datas.append(data)

    batch = Batch.from_data_list(datas)
    return batch