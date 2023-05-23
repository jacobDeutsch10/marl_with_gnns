import torch
from torch_geometric.data import Data, Batch
from sklearn.feature_extraction.image import grid_to_graph
from torch_geometric.utils import dense_to_sparse
import numpy as np

def random_mask_obs(obs, mask_prob=0.5):
    mask = torch.rand(obs.shape) < mask_prob
    obs[mask] = 0
    return obs

def get_adjacent(index, shape, hops=1):
    adjacent = []
    for c in range(shape[2]):
        for dx in range(-hops, hops+1):
            for dy in range(-hops, hops+1):
                if dx != 0 or dy != 0:  # exclude the cell itself
                    adjacent.append((index[0] + dx, index[1] + dy, c))
    valid_adjacent = [i for i in adjacent if 0 <= i[0] < shape[0] and 0 <= i[1] < shape[1]]
    return valid_adjacent
def gen_idx(shape):
    idx = np.array([
            [
                [ [i,j,k] for k in range(shape[2])]
            for j in range(shape[1])
        ] 
        for i in range(shape[0])
    ])
    
    return idx
def obs_to_graph_batch(obs):
    hops = 1
    datas = []
    for batch_id, observation in enumerate(obs):
        adj_matrix = grid_to_graph(*observation.shape, mask=observation.cpu().numpy(), return_as=np.ndarray)
        edge_index = dense_to_sparse(torch.as_tensor(adj_matrix, dtype=torch.long))[0].reshape(2, -1)
        # check dummy input and return valid index:
        if edge_index.shape[0] < 1:
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
        node_features = torch.as_tensor(gen_idx(observation.shape), dtype=torch.float).reshape(-1, 3)
        data = Data(x=node_features, edge_index=edge_index)
        datas.append(data)
    batch = Batch.from_data_list(datas)
    return batch