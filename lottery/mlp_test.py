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
from volterra_model_data import generate_data
from signals.utils import Figure, Subplot

import lott_lib


def main(_):
  console.start('vanilla_mlp')

  # Configurations
  MARK = 'mlp_vanilla_00'
  MEMORY_DEPTH = 30
  coe = 8
  HIDDEN_DIM = MEMORY_DEPTH * coe

  EPOCH = 5000
  LR = 0.000088
  BATCH_SIZE = 32
  PRINT_CYCLE = 10
  ACTIVATION = 'relu'

  FLAGS.train = True
  FLAGS.overwrite = True
  FLAGS.smart_train = True
  FLAGS.save_best = False
  FLAGS.summary = True
  # FLAGS.save_model = False
  FLAGS.snapshot = False
  FLAGS.epoch_tol = 100

  # Load data
  train_set, val_set, test_set = generate_data()
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Get model
  model = lott_lib.mlp00(MARK, MEMORY_DEPTH, HIDDEN_DIM, LR, ACTIVATION)

  # model.nn._branches_variables_assign(BRANCH_INDEX)

  # Train or evaluate
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=BATCH_SIZE,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH)
  else:
    model.evaluate(train_set, start_at=MEMORY_DEPTH)
    model.evaluate(val_set, start_at=MEMORY_DEPTH)
    model.evaluate(test_set, start_at=MEMORY_DEPTH, plot=True)

  console.end()


if __name__ == '__main__':
  tf.app.run()