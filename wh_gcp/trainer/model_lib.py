from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import Predictor
from tframe.models.sl.bamboo import Bamboo
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input
from tframe.layers.parametric_activation import Polynomial
from tframe.nets.resnet import ResidualNet
from tframe import pedia

from models.neural_net import NeuralNet


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

def mlp02(mark, memory_depth, layer_num, hidden_dim, learning_rate, activation):
  # Initiate a neural net
  model = NeuralNet(memory_depth, mark=mark, bamboo=True, identity_initial=True)
  nn = model.nn
  assert isinstance(nn, Bamboo)

  # Add layers
  nn.add(Input([memory_depth]))

  for _ in range(layer_num):
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

def svn_00(memory_depth, mark, hidden_dim, order1, order2, order3, learning_rate=0.001):

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

def svn_01(memory_depth, mark, hidden_dim, order1, learning_rate=0.001):

  strength = 0
  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial(order=order1))
  nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=strength, use_bias=False))

  # Build model
  nn.build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

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

# endregion : MLP
