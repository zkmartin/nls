from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import console
from tframe import Predictor
from tframe import TFData
from tframe.models.sl.vn import VolterraNet
from tframe.models.sl.bamboo import Bamboo
from tframe.models.sl.bamboo_broad import Bamboo_Broad
from tframe import pedia

from models import Model
from signals import Signal
from signals.utils.dataset import DataSet
from signals.utils.figure import Figure, Subplot
from signals.generator import gaussian_white_noise


class NeuralNet(Model):
  """A model for non-linear system based on neural network"""

  def __init__(self, memory_depth, mark='nn', degree=None, **kwargs):
    # Sanity check
    if memory_depth < 1: raise ValueError('!! Memory depth should be positive')

    # Call parent's construction methods
    Model.__init__(self)

    # Initialize fields
    self.memory_depth = memory_depth
    self.D = memory_depth
    self.degree = degree
    # TODO: compromise
    bamboo = kwargs.get('bamboo', False)
    bamboo_broad = kwargs.get('bamboo_broad', False)
    identity_inital = kwargs.get('identity_initial', False)
    if degree is not None:
      self.nn = VolterraNet(degree, memory_depth, mark, **kwargs)
    elif bamboo:
      self.nn = Bamboo(mark=mark, identity=identity_inital)
    elif bamboo_broad:
      self.nn = Bamboo_Broad(mark=mark, inter_type=pedia.fork, identity=identity_inital)
    else: self.nn = Predictor(mark=mark)

  # region : Public Methods

  def default_build(self, learning_rate=0.001, optimizer=None):
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer(learning_rate)
    self.nn.build(loss='euclid', metric='ratio', metric_name='Err %',
                  optimizer=optimizer)

  def inference(self, input_, **kwargs):
    if not self.nn.built:
      raise AssertionError('!! Model has not been built yet')
    mlp_input = self._gen_mlp_input(input_)
    tfinput = TFData(mlp_input)
    output = self.nn.predict(tfinput, **kwargs).flatten()

    output = Signal(output)
    output.__array_finalize__(input_)
    return output

  def identify(self, training_set, val_set=None, probe=None,
               batch_size=64, print_cycle=100, snapshot_cycle=1000,
               snapshot_function=None, epoch=1, **kwargs):
    # Train
    self.nn.train(batch_size=batch_size, training_set=training_set,
                  validation_set=val_set, print_cycle=print_cycle,
                  snapshot_cycle=snapshot_cycle, epoch=epoch, probe=None,
                  snapshot_function=snapshot_function, **kwargs)

  def evaluate(self, dataset, start_at=0, plot=False, **kwargs):
    # Check input
    if not isinstance(dataset, DataSet):
      raise TypeError('!! Input data set must be an instance of DataSet')
    if dataset.responses is None:
      raise ValueError('!! input data set should have responses')
    u, y = dataset.signls[0], dataset.responses[0]

    # Show status
    console.show_status('Evaluating {}'.format(dataset.name))

    # Evaluate
    system_output = y[start_at:]
    model_output = self(u, **kwargs)[start_at:]
    err = system_output - model_output
    ratio = lambda val: 100 * val / system_output.rms

    # The mean value of the simulation error in time domain
    val = err.average
    console.supplement('E[err] = {:.4f} ({:.3f}%)'.format(val, ratio(val)))
    # The standard deviation of the error in time domain
    val = float(np.std(err))
    console.supplement('STD[err] = {:.4f} ({:.3f}%)'.format(val, ratio(val)))
    # The root mean square value of the error in time domain
    val = err.rms
    console.supplement('RMS[err] = {:.6f} ({:.3f}%)'.format(val, ratio(val)))

    # Plot
    if not plot: return
    fig = Figure('Simulation Error')
    # Add ground truth
    prefix = 'System Output, $||y|| = {:.4f}$'.format(system_output.norm)
    fig.add(Subplot.PowerSpectrum(system_output, prefix=prefix))
    # Add model output
    prefix = 'Model Output, $||\Delta|| = {:.4f}$'.format(err.norm)
    fig.add(Subplot.PowerSpectrum(model_output, prefix=prefix, Error=err))
    # Plot
    fig.plot(ylim=True)


  def gen_snapshot_function(self, input_, response):
    from signals.utils import Figure, Subplot

    # Sanity check
    if not isinstance(input_, Signal) or not isinstance(response, Signal):
      raise TypeError('!! Input and response should be instances of Signal')

    def snapshot_function(obj):
      assert isinstance(obj, (Predictor, VolterraNet))
      pred = self(input_)
      delta = pred - response

      fig = Figure()
      fig.add(Subplot.PowerSpectrum(response, prefix='Ground Truth'))
      prefix = 'Predicted, $||\Delta||$ = {:.4f}'.format(delta.norm)
      fig.add(Subplot.PowerSpectrum(pred, prefix=prefix, Delta=delta))

      return fig.plot(show=False, ylim=True)

    return snapshot_function

  # endregion : Public Methods

  # region : Private Methods

  def _gen_mlp_input(self, input_):
    return input_.causal_matrix(self.memory_depth)

  # endregion : Private Methods


  """For some reason, do not remove this line"""


if __name__ == "__main__":
  model = NeuralNet(memory_depth=3, dont_build=True)
  input_ = np.arange(1, 11)
  print(input_)
  print(model._gen_mlp_input(input_))

