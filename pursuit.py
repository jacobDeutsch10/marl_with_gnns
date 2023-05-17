from pettingzoo.sisl import pursuit_v4, waterworld_v4
import ray
from ray import air, tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
import os 
import torch.nn as nn
from ray.rllib.models import ModelCatalog

from magnn.models import PursuitMLP, PursuitCNN, PursuitGNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", '--model', type=str, default="mlp", choices=["mlp", "cnn", "gnn"])


args = parser.parse_args()



model_map = {
    "mlp": "pursuitmlp",
    "cnn": "pursuitcnn",
    "gnn": "pursuitgnn"
}

os.environ["CUDA_VISIBLE_DEVICES"]="0"
ray.init(num_gpus=1, ignore_reinit_error=True)
register_env("pursuit", lambda _: PettingZooEnv(pursuit_v4.env()))

env = PettingZooEnv(pursuit_v4.env())
obs_space = env.observation_space
act_space = env.action_space
print(obs_space)
print(act_space)
print(env.reset())
ModelCatalog.register_custom_model(
        "pursuitmlp", PursuitMLP 
    )
ModelCatalog.register_custom_model(
        "pursuitcnn", PursuitCNN
)
ModelCatalog.register_custom_model(
        "pursuitgnn", PursuitGNN
)

tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episodes_total": 60000},
         local_dir="ray_results/pursuit",
        name=f"PPO_{args.model}_pursuit",
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=50,
        ),
    ),
    param_space={
        # Enviroment specific.
        "env": "pursuit",
        # General
        "framework": "torch",
        "batch_mode": "truncate_episodes",
        "num_gpus": 1,
        "num_workers": 6,
        "num_envs_per_worker": 4,
        "num_steps_sampled_before_learning_starts": 1000,
        "compress_observations": False,
        "rollout_fragment_length": 'auto',
        "train_batch_size": 5000,
        "model": { 'custom_model': model_map[args.model]},
        "gamma": 0.99,
        "n_step": 3,
        "lr": 0.001,
        "num_sgd_iter": 100,
        "sample_batch_size": 25,
        "sgd_minibatch_size": 256,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "lambda": 0.95,
        # Method specific.
        "multiagent": {
            # We only have one policy (calling it "shared").
            # Class, obs/act-spaces, and config will be derived
            # automatically.
            "policies": {"shared_policy"},
            "model": { 'custom_model': 'pursuitmpl'},
            # Always use "shared" policy.
            "policy_mapping_fn": (
                lambda agent_id, episode, worker, **kwargs: "shared_policy"
            ),
        },
    },
).fit()