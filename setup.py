from setuptools import setup, find_packages

setup(
    name='magnn',
    version='0.0.1',
    description='Multi-Agent Graph Neural Network',
    author='Jacob Deutsch and Mihir Kamble',
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "ray[rllib]",
        "pettingzoo",
        "gym",
        "networkx",
        "scipy",
        "scikit-learn",
        "tqdm",
        "tensorboard",
        "tensorflow_probability",
        "torch_geometric",
        "wandb",
        "pygame",
        "pymunk",
    ],
)