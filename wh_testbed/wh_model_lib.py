from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import FLAGS
from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Linear
from tframe.layers import Input
from tframe.layers.homogeneous import Homogeneous
from tframe.layers.parametric_activation import Polynomial
from tframe.layers.parametric_activation import Polynomial_Relu
from tframe.nets.resnet import ResidualNet
from tframe import pedia

from models.neural_net import NeuralNet

import layer_combs as lc


# region : Test

def test_00(memory, learning_rate=0.001):
  # Configurations
  mark = 'test'
  D = memory

  # Initiate model
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory]))

  nn.add(Linear(output_dim=2*D))
  nn.add(Activation('relu'))
  nn.add(Linear(output_dim=2*D))
  nn.add(Activation('relu'))
  nn.add(Linear(output_dim=2*D))
  nn.add(Polynomial(order=3))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : Test

# region : ResNet

def res_00(memory, blocks, order1, order2, activation='relu', learning_rate=0.001):
  # Configurations
  mark = 'res'
  D = memory

# Initiate model
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([D]))

  def add_res_block():
    net = nn.add(ResidualNet())
    net.add(Linear(output_dim=D))
    net.add(Activation(activation))
    net.add(Linear(output_dim=D))
    net.add_shortcut()
    net.add(Activation(activation))

  def add_res_block_poly():
    net = nn.add(ResidualNet())
    net.add(Linear(output_dim=D))
    net.add(Polynomial(order=order1))
    net.add(Linear(output_dim=D))
    net.add_shortcut()
    net.add(Polynomial(order=order2))

  if activation == 'poly':
    for _ in range(blocks):add_res_block_poly()
  else:
    for _ in range(blocks): add_res_block()

  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : ResNet

# region : SVN

def svn_00(memory, learning_rate=0.001):
  # Configuration
  D = memory
  hidden_dims = [2 * D] * 3
  p_order = 2
  mark = 'svn_{}_{}'.format(hidden_dims, p_order)

  # Initiate a predictor
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([D]))
  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim))
    nn.add(Polynomial(p_order))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  return model

def svn_01(memory_depth, mark, hidden_dim, order1, order2, order3, learning_rate=0.001):

  strength = 0
  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial(order=order1))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial(order=order2))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial(order=order3))
  nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=strength, use_bias=False))

  # Build model
  nn.build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

def svn_02(memory_depth, mark, hidden_dim, order1, order2, order3, learning_rate=0.001):

  strength = 0
  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial_Relu(order=order1))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial_Relu(order=order2))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial_Relu(order=order3))
  nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=strength, use_bias=False))

  # Build model
  nn.build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

# endregion : SVN

# region : NET

def net_00(memory_depth, learning_rate=0.001):
  # Configuration
  hidden_dim = 10
  homo_order = 4
  mark = 'net_h{}_homo{}'.format(hidden_dim, homo_order)

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(inter_type=pedia.sum)
  for i in range(1, homo_order + 1): nn.add_to_last_net(Homogeneous(i))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model


# endregion : NET

# region : MLP

def mlp_00(mark, memory_depth, layer_dim, layer_num, learning_rate,
           activation='relu'):
  # Configurations
  pass

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  for i in range(layer_num):
    nn.add(Linear(output_dim=layer_dim))
    nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

def mlp_01(mark, memory_depth, layer_dim, learning_rate,
           activation='relu'):
  # Configurations
  pass

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=layer_dim))
  nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : MLP

# region : Layers


# endregion : Utilities


"""LOGS
[1] mlp: 
    memory = 80
    loss = euclid
    hidden_dims = [160] * 4
    activation = lrelu

"""

