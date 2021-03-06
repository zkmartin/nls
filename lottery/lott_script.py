import tensorflow as tf
import os, sys

dn = os.path.dirname
abs_path = os.path.abspath(__file__)
sys.path.append(dn(abs_path))
sys.path.append(dn(dn(abs_path)))
del dn, abs_path

from tframe import console
from tframe import FLAGS

from signals.utils.dataset import load_wiener_hammerstein, DataSet

import lott_lib


def main(_):
  console.start('Lottery')

  # Configurations
  MARK = 'mlp00'
  MEMORY_DEPTH = 80
  coe = 8
  HIDDEN_DIM = MEMORY_DEPTH * coe

  EPOCH = 500
  LR = 0.000058
  BATCH_SIZE = 32
  PRINT_CYCLE = 10
  BRANCH_INDEX = 1
  FIX_PRE_WEIGHT = True
  ACTIVATION = 'relu'

  # FLAGS.train = False
  FLAGS.overwrite = True and BRANCH_INDEX == 0
  FLAGS.smart_train = True
  FLAGS.save_best = True and BRANCH_INDEX > 0
  FLAGS.summary = True
  # FLAGS.save_model = False
  FLAGS.snapshot = False
  FLAGS.epoch_tol = 50

  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    r'../data/wiener_hammerstein/whb.tfd', depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Get model
  model = lott_lib.mlp00(MARK, MEMORY_DEPTH, HIDDEN_DIM, LR, ACTIVATION)

  branch_1_weights = 'FeedforwardNet/branch/linear/weights:0'
  branch_1_bias = 'FeedforwardNet/branch/linear/biases:0'
  branch_2_weights = 'FeedforwardNet/branch2/linear/weights:0'
  branch_2_bias = 'FeedforwardNet/branch2/linear/biases:0'
  # model.nn.variable_assign(branch_1_weights, branch_2_weights)
  # model.nn.variable_assign(branch_1_bias, branch_2_bias)
  with model.nn._graph.as_default():
    variables = tf.trainable_variables()
    b = 1
  #   print(model.nn._session.run(variables[2]))
  #   print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #   print(model.nn._session.run(variables[6]))
  #   print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #   print(model.nn._session.run(variables[3]))
  #   print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #   print(model.nn._session.run(variables[7]))
  #   print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #   print(model.nn._session.run(variables[4]))
  #   print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #   print(model.nn._session.run(variables[5]))
  #   a = 1


  # Train or evaluate
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=BATCH_SIZE,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH,
                   branch_index=BRANCH_INDEX, freeze=FIX_PRE_WEIGHT)
  else:
    BRANCH_INDEX = 1
    model.evaluate(train_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(val_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(test_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)

  console.end()


if __name__ == '__main__':
  tf.app.run()
