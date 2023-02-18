Vehicle Routing Problem with Deep Reinforcement Learning
This project proposes a deep reinforcement learning (DRL) model to solve the Vehicle Routing Problem (VRP) with rich graph representation learning. Unlike existing Transformer-based DRL solutions that only consider node information, our model also incorporates edge information between nodes in the graph structure. We use a Transformer-based encoder-decoder framework with an edge information-embedded multi-head attention (EEMHA) layer in the encoder. The EEMHA-based encoder learns the underlying structure of the graph and generates an expressive graph topology representation by merging node and edge information. The model is trained using proximal policy optimization (PPO) with some code-level optimization techniques.

We conducted experiments on randomly generated instances and on real-world data generated from road networks to verify the performance of our proposed model. The results show that our model outperforms existing DRL methods and most conventional heuristics, with good generalizability from random instance training to real-world instance testing of different scales.

Dependencies
Python 3.7 or higher
PyTorch 1.7.1 or higher
TorchGeometric 1.5.0 or higher
Torch-Scatter 2.0.8 or higher
Torch-Sparse 0.6.12 or higher

Usage
The main script takes several command line arguments, including the input data file and various model hyperparameters. For example:
