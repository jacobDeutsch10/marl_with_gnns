from pettingzoo.sisl import pursuit_v4, waterworld_v4
import ray
from ray import air, tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
import os 
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.air.integrations.wandb import WandbLoggerCallback
from magnn.models import PursuitMLP, PursuitCNN, PursuitGNN, PursuitConvEncGNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", '--model', type=str, default="mlp", choices=["mlp", "cnn", "gnn"])

"""python pursuit.py -m mlp --train_batch_size 5000 --lr 0.0001 --batch_mode truncate_episodes"""


### PPO parameters
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--n_step", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_sgd_iter", type=int, default=5)
parser.add_argument("--sample_batch_size", type=int, default=25)
parser.add_argument("--sgd_minibatch_size", type=int, default=128)
parser.add_argument("--clip_param", type=float, default=0.1)
parser.add_argument("--vf_clip_param", type=float, default=10.0)
parser.add_argument("--entropy_coeff", type=float, default=0.01)
parser.add_argument("--kl_target", type=float, default=0.01)
parser.add_argument("--lambd", type=float, default=0.95)
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--num_envs_per_worker", type=int, default=4)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--compress_observations", type=bool, default=False)
parser.add_argument("--rollout_fragment_length", type=str, default='auto')
parser.add_argument("--train_batch_size", type=int, default=512)
parser.add_argument("--batch_mode", type=str, default="complete_episodes")
parser.add_argument("--framework", type=str, default="torch")
parser.add_argument("--use_wandb", type=bool, default=True)

args = parser.parse_args()



model_map = {
    "mlp": "pursuitmlp",
    "cnn": "pursuitcnn",
    "gnn": "pursuitgnn"
}

os.environ["CUDA_VISIBLE_DEVICES"]="0"
ray.init(num_gpus=1, ignore_reinit_error=True)
register_env("pursuit", lambda _: PettingZooEnv(pursuit_v4.env()))

ModelCatalog.register_custom_model(
        "pursuitmlp", PursuitMLP 
    )
ModelCatalog.register_custom_model(
        "pursuitcnn", PursuitCNN
)
ModelCatalog.register_custom_model(
        "pursuitgnn", PursuitConvEncGNN
)
cb = []
if args.use_wandb:
    cb = [WandbLoggerCallback(project="marl-w-gnn")]
tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episodes_total": 600},
         local_dir="ray_results/pursuit",
        name=f"PPO_{args.model}_pursuit",
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=50,
        ),
        callbacks=cb,
    ),
    param_space={
        # Enviroment specific.
        "env": "pursuit",
        # General
        "framework": "torch",
        "batch_mode": "complete_episodes",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "compress_observations": args.compress_observations,
        "rollout_fragment_length": args.rollout_fragment_length,
        "train_batch_size": args.train_batch_size,
        "model": { 'custom_model': model_map[args.model]},
        "gamma": args.gamma,
        "n_step": args.n_step,
        "lr": args.lr,
        "num_sgd_iter": args.num_sgd_iter,
        "sample_batch_size": args.sample_batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "clip_param": args.clip_param,
        "vf_clip_param": args.vf_clip_param,
        "entropy_coeff": args.entropy_coeff,
        "kl_target": args.kl_target,
        "lambda": args.lambd,
        # Method specific.
        "multiagent": {
            # We only have one policy (calling it "shared").
            # Class, obs/act-spaces, and config will be derived
            # automatically.
            "policies": {"shared_policy"},
            "model": { 'custom_model': model_map[args.model]},
            # Always use "shared" policy.
            "policy_mapping_fn": (
                lambda agent_id, episode, worker, **kwargs: "shared_policy"
            ),
        },
    },
).fit()