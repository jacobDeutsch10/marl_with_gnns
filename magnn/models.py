
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
import time
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential, GENConv
from torch_geometric.nn import global_mean_pool
from magnn.transforms import obs_to_graph_batch, random_mask_obs
from torch_geometric.data import Data, Batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PursuitGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.mask_prob = model_config["custom_model_config"]["mask_prob"]

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
        model_in = obs_to_graph_batch(obs).to(device)
        self._model_out = self.model(model_in.x, model_in.edge_index, model_in.batch)
        policy_out = self.policy_fn(self._model_out)
        return policy_out, []

class LearnedThreshold(nn.Module):
    def __init__(self):
        super(LearnedThreshold, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Initialize threshold as 0.5

    def forward(self, x):
        return (x > self.threshold).float() * 1.0

class PursuitConvEncGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, [3,3], stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, [3, 3], stride=1, padding=1),
            nn.ReLU(),
        )
        self.model = Sequential('x, edge_index, edge_weight, batch', [
            (GCNConv(32, 64), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 128), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(128, 128), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(inplace=True),
            (global_mean_pool, ('x, batch -> x'))
        ])

        self.value_fn = nn.Linear(128, 1) 
        self.policy_fn = nn.Linear(128, num_outputs)
    def value_function(self):
        value_out = self.value_fn(self._model_out)
        return torch.reshape(value_out, [-1])
    def encode_obs(self, obs):
        obs_enc = self.encoder(obs.permute(0, 3, 1, 2))
        x = torch.diagonal(obs_enc, dim1=2, dim2=3).permute(0, 2, 1)
        edge_weight = obs_enc.mean(dim=1)
        non_zero = torch.nonzero(edge_weight)
        if non_zero.shape[0] == 0:
            return None, None, None, None

        batch = non_zero[:, 0]
        edge_index = torch.cat((non_zero[:, 1].unsqueeze(0)+147*batch, non_zero[:, 2].unsqueeze(0)+147*batch), dim=0)
        edge_attr = x[non_zero[:,0] :, non_zero[:,1], non_zero[:,2]]

        return x, edge_index, edge_attr, batch



    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x, edge_index, edge_attr, batch = self.encode_obs(input_dict["obs"])
        if x is None:
            self._model_out = torch.zeros((input_dict["obs"].shape[0], 128)).to(device)
            return torch.zeros((input_dict["obs"].shape[0], self.num_outputs)).to(device), []
        self._model_out = self.model(x, edge_index, edge_attr, batch)
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
    

class MultiAgentGraph(nn.Module):
    def __init__(self, num_landmarks, num_agents):
        super(MultiAgentGraph, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_agents = num_agents

    def forward(self, batch_observations):
        batch_data = []
        for obs in batch_observations:
            agent_vel_pos = obs[:4]  # velocity and position
            rel_landmarks = obs[4:4+2*self.num_landmarks].reshape(self.num_landmarks, 2)
            rel_others = obs[4+2*self.num_landmarks:4+2*(self.num_landmarks+self.num_agents-1)].reshape(self.num_agents-1, 2)
            comm = obs[-self.num_agents+1:].reshape(self.num_agents-1, 1)

            # Absolute positions
            abs_landmarks = agent_vel_pos[2:4] + rel_landmarks
            abs_others = agent_vel_pos[2:4] + rel_others
            landmark_features = torch.concatenate((
                abs_landmarks, 
                rel_landmarks, 
                rel_landmarks/(1e-3+agent_vel_pos[:2])
                , torch.zeros(abs_landmarks.shape[0], 2)), dim=1)
            other_features = torch.concatenate((
                abs_others,
                rel_others,
                rel_others/(1e-3+agent_vel_pos[:2]),
                torch.ones(abs_others.shape[0], 1),
                comm), dim=1)
            agent_features = torch.concatenate((
                agent_vel_pos[2:4].reshape(1, 2),
                agent_vel_pos[:2].reshape(1, 2),
                torch.zeros(1, 2),
                torch.ones(1, 1)*2,
                torch.zeros(1, 1)), dim=1)
            # Concatenate all node features
            x = torch.cat((agent_features, landmark_features, other_features), dim=0)

            # Each node is connected with every other node
            edge_index = torch.combinations(torch.arange(self.num_agents + self.num_landmarks), r=2).t()

            # Create edge attributes: Euclidean distance between nodes
            node_features = x.reshape(self.num_agents + self.num_landmarks, -1)
            edge_attr = (node_features[edge_index[0]][:2] - node_features[edge_index[1]][:2]).pow(2).sum(-1).sqrt()

            data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
            batch_data.append(data)

        batch = Batch.from_data_list(batch_data).to(device)
        return batch
    
class SpreadGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.to_graph = MultiAgentGraph(3, 3)
        self.model = Sequential('x, edge_index, batch', [
            (GENConv(3, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GENConv(64, 128), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GENConv(128, 128), 'x, edge_index -> x'),
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
        with torch.no_grad():
            model_in = self.to_graph(input_dict["obs"].float()).to(device) 
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