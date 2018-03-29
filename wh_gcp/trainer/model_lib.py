from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import Predictor
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input
from tframe.layers.parametric_activation import Polynomial

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

# endregion : MLP
