import numpy as np


class Loss:
    """Base class for loss functions"""
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    """Categorical cross entropy loss. Used for multiclass classifiers"""
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # Clip gradients to prevent exploding

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
                ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
                )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

    def backward_with_softmax(self, dvalues, y_true):
        samples = len(dvalues)
        labels = dvalues.shape[1]

      # If y_true is class indices, one-hot encode
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = (dvalues - y_true) / samples
