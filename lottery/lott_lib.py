from models import NeuralNet
import tensorflow as tf

from tframe.models.sl.bamboo import Bamboo

from tframe.layers import Activation
from tframe.layers import Input
from tframe.layers import Linear


def mlp00(mark, memory_depth, hidden_dim, learning_rate, activation):
  # Initiate a neural net
  model = NeuralNet(memory_depth, mark=mark, bamboo=True)
  nn = model.nn
  assert isinstance(nn, Bamboo)

  # Add layers
  nn.add(Input([memory_depth]))

  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  branch = nn.add_branch()
  branch.add(Linear(output_dim=1))

  nn.add(Linear(output_dim=hidden_dim, weight_initializer='identify', bias_initializer=tf.zeros_initializer()))
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

