import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exps = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exps / np.sum(exps, axis=-1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output - self.output * np.sum(dvalues * self.output, axis=-1, keepdims=True)
