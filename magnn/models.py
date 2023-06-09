
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, GENConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from magnn.transforms import obs_to_graph_batch, random_mask_obs, GraphTransformer
from torch_geometric.data import Data, Batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PursuitGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.mask_prob = model_config["custom_model_config"]["mask_prob"]
        self.graph_transformer = GraphTransformer(obs_space.shape[0])
        self.model = Sequential('x, edge_index, batch', [
            (GCNConv(3, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 128), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(128, 128), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, ('x, batch -> x'))
        ])

        self.value_fn = nn.Linear(128, 1) 
        self.policy_fn = nn.Linear(128, num_outputs)
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = random_mask_obs(input_dict["obs"].float(), self.mask_prob)
        model_in = self.graph_transformer.transform(obs).to(device)
        self._model_out = self.model(model_in.x, model_in.edge_index, model_in.batch)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []
    
class PursuitGAT(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.mask_prob = model_config["custom_model_config"]["mask_prob"]
        self.graph_transformer = GraphTransformer(obs_space.shape[0])
        self.model = Sequential('x, edge_index, batch', [
            (GATConv(3, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(64, 64), 'x, edge_index -> x'),
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
        obs = random_mask_obs(input_dict["obs"].float(), self.mask_prob)
        model_in = self.graph_transformer.transform(obs).to(device)
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
    

class SpreadMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        x = obs_space.shape[0]
        self.model = nn.Sequential(
            nn.Linear(x, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            
        )
        self.policy_fn = nn.Linear(128, num_outputs)
        self.value_fn = nn.Linear(128, 1)
        self._model_out = None
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_in = input_dict["obs"].float()
        self._model_out = self.model(model_in)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []
    
class SpreadGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        self.num_units = obs_space.shape[0]//5
        nn.Module.__init__(self)
        self.model = Sequential('x, edge_index, batch', [
            (SAGEConv(5, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (SAGEConv(64, 128), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (SAGEConv(128, 128), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, ('x, batch -> x'))
        ])
        self.edge_index = torch.zeros((self.num_units, self.num_units)).nonzero().t()
        self.value_fn = nn.Linear(128, 1) 
        self.policy_fn = nn.Linear(128, num_outputs)
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_in = input_dict["obs"].float().reshape(-1, self.num_units, 5)
        
        
        datas = []
        for i in range(model_in.shape[0]):
            datas.append(
                Data(
                    x=model_in[i],
                    edge_index=self.edge_index,
                )
            )
        model_in = Batch.from_data_list(datas).to(device)
        self._model_out = self.model(model_in.x, model_in.edge_index, model_in.batch)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []

class PursuitMLP(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.mask_prob = model_config["custom_model_config"]["mask_prob"]
        x, y, c = obs_space.shape
        self.model = nn.Sequential(
            nn.Linear(x*y*c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            
        )
        self.policy_fn = nn.Linear(128, num_outputs)
        self.value_fn = nn.Linear(128, 1)
        self._model_out = None
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = random_mask_obs(input_dict["obs"].float(), self.mask_prob)
        model_in = obs.flatten(1)
        self._model_out = self.model(model_in)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []