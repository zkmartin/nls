from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet

import wh_model_lib

# =============================================================================
# Global configuration
WH_PATH = '../data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000

coe = 80
MEMORY_DEPTH = 80
D = MEMORY_DEPTH
NN_EPOCH = 30
NN_HID_DIMS = [D*coe] * 1
NN_LEARNING_RATE = 0.00008
BATCH_SIZE = 32
PRINT_CYCLE = 10

FLAGS.train = True
# FLAGS.train = False
FLAGS.overwrite = True
# FLAGS.overwrite = False
FLAGS.save_best = True
# FLAGS.save_best = True

FLAGS.smart_train = False
FLAGS.epoch_tol = 50

# Turn off overwrite while in save best mode
FLAGS.overwrite = FLAGS.overwrite and not FLAGS.save_best and FLAGS.train

EVALUATION = not FLAGS.train
PLOT = EVALUATION
# =============================================================================
# Load data set
train_set, val_set, test_set = load_wiener_hammerstein(
  WH_PATH, depth=MEMORY_DEPTH)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)
assert isinstance(test_set, DataSet)

model = wh_model_lib.mlp_00(MEMORY_DEPTH, NN_HID_DIMS, NN_LEARNING_RATE)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=BATCH_SIZE,
  print_cycle=PRINT_CYCLE, epoch=NN_EPOCH)

# Evaluation
if EVALUATION:
  model.evaluate(train_set, start_at=MEMORY_DEPTH, plot=False)
  model.evaluate(val_set, start_at=MEMORY_DEPTH, plot=False)
  model.evaluate(test_set, start_at=MEMORY_DEPTH, plot=PLOT)
