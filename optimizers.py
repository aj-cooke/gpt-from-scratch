import numpy as np


class Optimizer_SGD:
  """
  Optimize neural network with stochastic gradient descent.
  Works with dense layers, embedding, attention, and layernorm.
  """
  def __init__(self, learning_rate=1., decay=0., momentum=0.):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum

  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * ( 1. / (1. + self.decay * self.iterations))

  def update_params(self, layer):
    self.update_weights(layer)
    self.update_biases(layer)

  def update_weights(self, layer):
    if self.momentum:

      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)

      weight_updates = \
        self.momentum * layer.weight_momentums - \
        self.current_learning_rate * layer.dweights
      layer.weight_momentums = weight_updates

    else:
      weight_updates = -self.current_learning_rate * \
                        layer.dweights

    layer.weights += weight_updates

  def update_attention(self, layer):
    if self.momentum:

      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = True
        layer.Qw_momentums = np.zeros_like(layer.Qw)
        layer.Kw_momentums = np.zeros_like(layer.Kw)
        layer.Vw_momentums = np.zeros_like(layer.Vw)

      Qw_updates = \
        self.momentum * layer.Qw_momentums - \
        self.current_learning_rate * layer.dQw
      layer.Qw_momentums = Qw_updates

      Kw_updates = \
        self.momentum * layer.Kw_momentums - \
        self.current_learning_rate * layer.dKw
      layer.Kw_momentums = Kw_updates

      Vw_updates = \
        self.momentum * layer.Vw_momentums - \
        self.current_learning_rate * layer.dVw
      layer.Vw_momentums = Vw_updates

    else:
      Qw_updates = -self.current_learning_rate * \
                        layer.dQw
      Kw_updates = -self.current_learning_rate * \
                        layer.dKw
      Vw_updates = -self.current_learning_rate * \
                        layer.dVw

    layer.Qw += Qw_updates
    layer.Kw += Kw_updates
    layer.Vw += Vw_updates

  def update_biases(self, layer):
      if self.momentum:
          if not hasattr(layer, 'bias_momentums'):
              layer.bias_momentums = np.zeros_like(layer.biases)

          bias_updates = self.momentum * layer.bias_momentums - \
                        self.current_learning_rate * layer.dbiases
          layer.bias_momentums = bias_updates
      else:
          bias_updates = -self.current_learning_rate * layer.dbiases

      layer.biases += bias_updates

  def update_layernorm(self, layer):
    if self.momentum:

        if not hasattr(layer, 'gamma_momentums'):
            layer.gamma_momentums = np.zeros_like(layer.gamma)
            layer.beta_momentums = np.zeros_like(layer.beta)

        gamma_updates = self.momentum * layer.gamma_momentums - \
                        self.current_learning_rate * layer.dgamma
        beta_updates = self.momentum * layer.beta_momentums - \
                       self.current_learning_rate * layer.dbeta

        layer.gamma_momentums = gamma_updates
        layer.beta_momentums = beta_updates

    else:
        gamma_updates = -self.current_learning_rate * layer.dgamma
        beta_updates = -self.current_learning_rate * layer.dbeta

    layer.gamma += gamma_updates
    layer.beta += beta_updates

  def post_update_params(self):
    self.iterations += 1
