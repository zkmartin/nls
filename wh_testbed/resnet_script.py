from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet

import wh_model_lib

# =============================================================================
# Global configuration
WH_PATH = '../data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000

MEMORY_DEPTH = 80
D = MEMORY_DEPTH*16
NN_EPOCH = 300
NN_BLOCKS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
PRINT_CYCLE = 10
blocks = 2
order1 = 4
order2 = 4
coe = 16
LAYER_DIM = MEMORY_DEPTH*coe
ACTIVATION = 'poly'


FLAGS.train = True
# FLAGS.train = False
# FLAGS.overwrite = True
FLAGS.overwrite = True
# FLAGS.save_best = False
FLAGS.save_best = False

FLAGS.smart_train = True
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

model = wh_model_lib.res_00(memory=MEMORY_DEPTH, blocks=NN_BLOCKS, order1=order1, order2=order2,
                         activation=ACTIVATION, learning_rate=LEARNING_RATE)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=BATCH_SIZE,
  print_cycle=PRINT_CYCLE, epoch=NN_EPOCH)

# Evaluation
if EVALUATION:
  model.evaluate(train_set, start_at=1000, plot=False)
  model.evaluate(val_set, start_at=80, plot=False)
  model.evaluate(test_set, start_at=1000, plot=False)
