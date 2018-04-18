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
  console.start('Lottery')

  # Configurations
  MARK = 'mlp_volterra_test0'
  MEMORY_DEPTH = 30
  coe = 8
  HIDDEN_DIM = MEMORY_DEPTH * coe
  BRANCH_NUM = 10
  T_BRANCH_INDEX_S = 0
  T_BRANCH_INDEX_E = 2

  EPOCH = 5000
  LR = 0.00000088
  LR_LIST = [0.000088]*(BRANCH_NUM + 1)
  BATCH_SIZE = 32
  PRINT_CYCLE = 10
  BRANCH_INDEX = 1
  FIX_PRE_WEIGHT = False
  ACTIVATION = 'relu'

  FLAGS.train = False
  FLAGS.overwrite = True and BRANCH_INDEX == 0
  FLAGS.smart_train = True
  FLAGS.save_best = True and BRANCH_INDEX > 0
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
  model = lott_lib.mlp01(MARK, MEMORY_DEPTH, BRANCH_NUM, HIDDEN_DIM, LR, ACTIVATION)

  # model.nn._branches_variables_assign(BRANCH_INDEX)

  # Train or evaluate
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=BATCH_SIZE,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH,
                   branch_index=BRANCH_INDEX, lr_list=LR_LIST,
                   freeze=FIX_PRE_WEIGHT, t_branch_s_index=T_BRANCH_INDEX_S,
                   t_branch_e_index=T_BRANCH_INDEX_E,
                   layer_train=True)

  else:
    BRANCH_INDEX = 2
    model.evaluate(train_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(val_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(test_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX, plot=True)

  console.end()


if __name__ == '__main__':
  tf.app.run()