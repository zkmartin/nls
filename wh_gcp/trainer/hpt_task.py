import tensorflow as tf

from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib

# Add command-line arguments for hyper-parameters
flags = tf.app.flags

flags.DEFINE_integer("layer_num", 3, "Layer number")
flags.DEFINE_integer("coe", 8, "the coe of layer_dim")
flags.DEFINE_string("activation", 'relu', "the activation of the layer")
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
  FLAGS.smart_train = True
  FLAGS.save_best = True

  MARK = 'mlp00'
  MEMORY_DEPTH = 80
  PRINT_CYCLE = 50
  EPOCH = 300

  LAYER_DIM = MEMORY_DEPTH * FLAGS.coe
  LAYER_NUM = FLAGS.layer_num
  LEARNING_RATE = FLAGS.lr
  BATCH_SIZE = 32
  ACTIVATION = FLAGS.activation

  # Get model
  model = model_lib.mlp_00(
    MARK, MEMORY_DEPTH, LAYER_DIM, LAYER_NUM, LEARNING_RATE, ACTIVATION)

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
