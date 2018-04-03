import tensorflow as tf

from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib

# Add command-line arguments for hyper-parameters
flags = tf.app.flags

flags.DEFINE_integer("coe", 2, "layer_dim coe")
flags.DEFINE_integer("blocks", 1, "the blocks of resnet")
flags.DEFINE_integer("order1", 2, "the order of polynomial")
flags.DEFINE_integer("order2", 2, "the order of polynomial")
flags.DEFINE_string("activation", 'relu', "activation function")

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

  MARK = 'resnet01'
  MEMORY_DEPTH = 80
  PRINT_CYCLE = 50
  EPOCH = 100
  NN_BLOCKS = FLAGS.blocks
  order1 = FLAGS.order1
  order2 = FLAGS.order2


  LAYER_DIM = MEMORY_DEPTH * FLAGS.coe
  LEARNING_RATE = FLAGS.lr
  ACTIVATION = FLAGS.activation


  # Get model
  model = model_lib.res_00(memory=MEMORY_DEPTH, blocks=NN_BLOCKS, order1=order1, order2=order2,
                           activation=ACTIVATION, learning_rate=LEARNING_RATE)

  # Load data set
  train_set, val_set, test_set = load_wiener_hammerstein(
    WH_PATH, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=64,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH)

  console.end()


if __name__ == "__main__":
  tf.app.run()