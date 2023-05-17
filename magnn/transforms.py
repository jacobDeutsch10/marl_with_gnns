import torch
from torch_geometric.data import Data, Batch


def get_adjacent(index, shape, hops=1):
    adjacent = []
    for c in range(shape[2]):
        for dx in range(-hops, hops+1):
            for dy in range(-hops, hops+1):
                if dx != 0 or dy != 0:  # exclude the cell itself
                    adjacent.append((index[0] + dx, index[1] + dy, c))
    valid_adjacent = [i for i in adjacent if 0 <= i[0] < shape[0] and 0 <= i[1] < shape[1]]
    return valid_adjacent

def obs_to_graph_batch(obs):
    hops = 1
    datas = []
    for batch_id, observation in enumerate(obs):
        nodes = []
        edge_index = []
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                for k in range(observation.shape[2]):
                    if observation[i, j, k] > 0:
                        nodes.append((batch_id, (i, j, k), k))
        node_dict = {node[1]: idx for idx, node in enumerate(nodes)}
        for node in nodes:
            adjacents = get_adjacent(node[1], observation.shape, hops=hops)
            for adj in adjacents:
                if adj in node_dict:
                    edge_index.append((node_dict[node[1]], node_dict[adj]))
        if len(node_dict) == 0:
            # do this so we dont error on dummy batch
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
            node_features = torch.tensor([[0,1,2]], dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            node_features = torch.tensor([node[1] for node in nodes], dtype=torch.float)
        data = Data(x=node_features.reshape(-1,3), edge_index=edge_index)
        datas.append(data)

    batch = Batch.from_data_list(datas)
    return batch