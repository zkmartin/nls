from models import NeuralNet
import tensorflow as tf

from tframe.models.sl.bamboo import Bamboo
from tframe.models.sl.bamboo_broad import Bamboo_Broad
from tframe.nets.resnet import ResidualNet

from tframe.layers import Activation
from tframe.layers import Input
from tframe.layers import Linear
from tframe.layers import Dropout


def mlp00(mark, memory_depth, hidden_dim, learning_rate, activation):
  # Initiate a neural net
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn

  # Add layers
  nn.add(Input([memory_depth]))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate=learning_rate)

  # Return model
  return model

def mlp01(mark, memory_depth, branch_num, hidden_dim, learning_rate, activation, identity_init=True):
  # Initiate a neural net
  if identity_init:
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

def mlp02(mark, memory_depth, branch_num, hidden_dim, learning_rate, activation, identity_init=False):
  # Initiate a neural net
  if identity_init:
    model = NeuralNet(memory_depth, mark=mark, bamboo_braod=True, identity_initial=True)
  else:
    model = NeuralNet(memory_depth, mark=mark, bamboo_broad=True, identity_initial=False)

  nn = model.nn
  assert isinstance(nn, Bamboo_Broad)

  # Add layers
  nn.add(Input([memory_depth]))

  branch = nn.add_branch()
  branch.add(Linear(output_dim=hidden_dim))
  branch.add(Activation(activation))
  branch.add(Linear(output_dim=1))

  for _ in range(branch_num - 1):
    branch = nn.add_branch()
    branch.add(Linear(output_dim=hidden_dim, weight_initializer=tf.zeros_initializer(),
                      bias_initializer=tf.zeros_initializer()))
    branch.add(Activation(activation))
    branch.add(Linear(output_dim=1, weight_initializer=tf.zeros_initializer(),
                      bias_initializer=tf.zeros_initializer()))

  # Build model
  model.default_build(learning_rate)



  # Return model
  return model

def mlp_res00(mark, memory_depth, branch_num, hidden_dim, learning_rate, activation, identity_init=True):
  # Initiate a neural net
  if identity_init:
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

  resnet = nn.add(ResidualNet())
  resnet.add(Linear(output_dim=hidden_dim))
  resnet.add(Activation(activation))
  resnet.add(Linear(output_dim=hidden_dim))
  resnet.add_shortcut()
  resnet.add(Activation(activation))

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
