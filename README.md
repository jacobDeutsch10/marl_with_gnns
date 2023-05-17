# MARL with GNNs

## Stanford CS224r Final Project
by Jacob Deutsch and Mihir Kamble

## Install:
```bash
git clone git@github.com:jacobDeutsch10/marl_with_gnns.git

cd marl_with_gnns

pip install -e .

wandb login # copy api key
```

## run

```bash
# run pursuit with MLP
python pursuit.py -m mlp

# CNN
python pursuit.py -m cnn

# GNN
python pursuit.py -m gnn

```
