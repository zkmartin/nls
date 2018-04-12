from models import NeuralNet
import tensorflow as tf

from tframe.models.sl.bamboo import Bamboo

from tframe.layers import Activation
from tframe.layers import Input
from tframe.layers import Linear


def mlp00(mark, memory_depth, hidden_dim, learning_rate, activation):
  # Initiate a neural net
  model = NeuralNet(memory_depth, mark=mark, bamboo=True, identify_intial=True)
  nn = model.nn
  assert isinstance(nn, Bamboo)

  # Add layers
  nn.add(Input([memory_depth]))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  branch = nn.add_branch()
  branch.add(Linear(output_dim=1))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  branch = nn.add_branch()
  branch.add(Linear(output_dim=1))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate=learning_rate)

  # Return model
  return model

def mlp01(mark, memory_depth, branch_num, hidden_dim, learning_rate, activation, identiry_init=True):
  # Initiate a neural net
  if identiry_init:
    model = NeuralNet(memory_depth, mark=mark, bamboo=True, identity_initial=True)
  else:
    model = NeuralNet(memory_depth, mark=mark, bamboo=True, identity_initial=False)
  nn = model.nn
  assert isinstance(nn, Bamboo)

  # Add layers
  nn.add(Input([memory_depth]))

  for _ in range(branch_num):
    nn.add(Linear(output_dim=hidden_dim))
    nn.add(Activation(activation))
    branch = nn.add_branch()
    branch.add(Linear(output_dim=1))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)


  # Return model
  return model

if __name__ == '__main__':
  lr_list = [0.001, 0.001, 0.001]
  model = mlp01('test', 10, 10, 0.001, 'relu')
