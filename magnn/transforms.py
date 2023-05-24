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
        for dx in range(0, hops+1):
            for dy in range(0, hops+1):
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
    obs_shape = obs.shape[1:]
    original_shape = (int(np.sqrt(obs_shape[0]//3)), int(np.sqrt(obs_shape[0]//3)), 3) # guess the original shape
    for batch_id, observation in enumerate(obs):
        adj_matrix = observation
        edge_index = dense_to_sparse(torch.as_tensor(adj_matrix, dtype=torch.long))[0].reshape(2, -1)
        # check dummy input and return valid index:
        if edge_index.shape[0] < 1:
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
        node_features = torch.as_tensor(gen_idx(original_shape), dtype=torch.float).reshape(-1, 3)
        data = Data(x=node_features, edge_index=edge_index)
        datas.append(data)
    batch = Batch.from_data_list(datas)
    return batch

class GraphTransformer():
    def __init__(self, obs_size):
        self.nodes = []
        for i in range(obs_size):
            for j in range(obs_size):
                for k in range(3):
                    self.nodes.append((i, j, k))
        node_dict = {node: idx for idx, node in enumerate(self.nodes)}
        self.node_dict = node_dict
        self.node_feats = torch.as_tensor(self.nodes, dtype=torch.float)
        self.obs_size = obs_size

    def transform(self, batch_obs):
        datas = []
        for obs in batch_obs:
            adj_t = np.zeros((len(self.nodes), len(self.nodes)))
            for node in self.nodes:
                adjacents = get_adjacent(node, obs.shape)
                for adj in adjacents:
                    
                    adj_t[self.node_dict[node], self.node_dict[adj]] = 1
                    adj_t[self.node_dict[adj], self.node_dict[node]] = 1
            edge_index = dense_to_sparse(torch.as_tensor(adj_t, dtype=torch.long))[0].reshape(2, -1)
            if edge_index.shape[0] < 1:
                edge_index = torch.tensor([[0, 0], [0, 1], [1, 2]], dtype=torch.long).t().contiguous()
            data = Data(x=self.node_feats, edge_index=edge_index, num_nodes=len(self.nodes))
            datas.append(data)
        batch = Batch.from_data_list(datas)
        return batch