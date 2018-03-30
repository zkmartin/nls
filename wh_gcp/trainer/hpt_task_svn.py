import tensorflow as tf

from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib

# Add command-line arguments for hyper-parameters
flags = tf.app.flags

flags.DEFINE_integer("order1", 2, "order1")
flags.DEFINE_integer("order2", 2, "order2")
flags.DEFINE_integer("order3", 2, "order3")
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

  WH_PATH = FLAGS.data_dir

  MARK = 'svn00'
  MEMORY_DEPTH = 80
  PRINT_CYCLE = 50
  EPOCH = 100

  LAYER_DIM = MEMORY_DEPTH * 2
  LEARNING_RATE = FLAGS.lr
  BATCH_SIZE = 64
  ORDER1 = FLAGS.order1
  ORDER2 = FLAGS.order2
  ORDER3 = FLAGS.order3


  # Get model
  model = model_lib.svn_00(
    MEMORY_DEPTH, MARK, LAYER_DIM, ORDER1, ORDER2, ORDER3, LEARNING_RATE)
  # Load data set
  train_set, val_set, test_set = load_wiener_hammerstein(
    WH_PATH, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=BATCH_SIZE,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH)


  console.end()


if __name__ == "__main__":
  tf.app.run()