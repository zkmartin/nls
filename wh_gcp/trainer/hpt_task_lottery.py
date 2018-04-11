import tensorflow as tf

from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib

# Add command-line arguments for hyper-parameters
flags = tf.app.flags

flags.DEFINE_integer("coe", 2, "layer_dim coe")
flags.DEFINE_integer("branches", 2, "the branches of the model")
flags.DEFINE_string("activation", 'relu', "activation function")
flags.DEFINE_float('lr1', 0.0001, 'the lr for the first layer')
flags.DEFINE_float('lr2', 0.0001, 'the lr for the second layer')
flags.DEFINE_float('lr3', 0.0001, 'the lr for the third layer')
flags.DEFINE_float('lr4', 0.0001, 'the lr for the fourth layer')
flags.DEFINE_float('lr5', 0.0001, 'the lr for the fifth layer')
#flags.DEFINE_float("lr", 0.001, "Learning rate")
#flags.DEFINE_integer("batch_size", -1, "The size of batch images")

FLAGS = flags.FLAGS

def main(_):
  console.start('trainer.task')

  # Set default flags
  FLAGS.train = True
  if FLAGS.use_default:
    FLAGS.overwrite = True
    FLAGS.smart_train = False
    FLAGS.save_best = False

  FLAGS.smart_train = True
  FLAGS.save_best = True

  WH_PATH = FLAGS.data_dir

  MARK = 'lottery01'
  MEMORY_DEPTH = 80
  PRINT_CYCLE = 50
  EPOCH = 500
  LR = 0.000088


  LAYER_DIM = MEMORY_DEPTH * FLAGS.coe
  LR_LIST = [FLAGS.lr1, FLAGS.lr2, FLAGS.lr3, FLAGS.lr4, FLAGS.lr5]
  ACTIVATION = FLAGS.activation
  BRANCHES = FLAGS.branches
  FLAGS.smart_train = True

  # Get model
  model = model_lib.mlp02(MARK, MEMORY_DEPTH, BRANCHES, LAYER_DIM, LR, ACTIVATION)

  # Load data set
  train_set, val_set, test_set = load_wiener_hammerstein(
    WH_PATH, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=64,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH,
                   lr_list=LR_LIST)

  console.end()


if __name__ == "__main__":
  tf.app.run()