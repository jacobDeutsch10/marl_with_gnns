
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn import global_mean_pool
from magnn.transforms import obs_to_graph_batch
from torch_geometric.data import Data, Batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PursuitGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.model = Sequential('x, edge_index, batch', [
            (GCNConv(3, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, ('x, batch -> x'))
        ])

        self.value_fn = nn.Linear(64, 1) 
        self.policy_fn = nn.Linear(64, num_outputs)
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        edge_indices, node_features = obs["edge_index"].unbatch_all(), obs["node_features"].unbatch_all()
        # check for dummy data
        if len(edge_indices[0]) == 0:
            edge_indices = [torch.tensor([[0, 0], [1, 1]]).to(device).T for _ in range(len(edge_indices))]
        if len(node_features[0]) == 0:
            node_features = [torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.float).to(device) for _ in range(len(node_features))]
        model_in = Batch.from_data_list([Data(x=node_features[i], edge_index=edge_indices[i].T) for i in range(len(node_features))])
        self._model_out = self.model(model_in.x, model_in.edge_index, model_in.batch)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []


class PursuitCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [2,2], stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, [2, 2], stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, [2, 2], stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(3136, num_outputs)
        self.value_fn = nn.Linear(3136, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()
    

class PursuitMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        x, y, c = obs_space.shape
        self.model = nn.Sequential(
            nn.Linear(x*y*c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )
        self.v_model = nn.Sequential(
            nn.Linear(x*y*c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self._model_in = None
    def value_function(self):
        value_out = self.v_model(self._model_in)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._model_in = input_dict["obs"].float().flatten(1)
        model_out = self.model(self._model_in)
        return model_out, []