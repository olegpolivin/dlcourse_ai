import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss, grad = reg_strength*np.sum(W**2), 2*reg_strength*W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    # Done below
    if len(predictions.shape) == 1:
      # batch_size of 1
      probs = predictions - np.max(predictions)
      probs = np.exp(probs) / np.sum(np.exp(probs))
    elif len(predictions.shape) > 1:
      probs = predictions - np.max(predictions, axis=1)[:, np.newaxis]
      probs = np.exp(probs) / np.sum(np.exp(probs), axis=1)[:, np.newaxis]
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    # Done below
    if len(probs.shape) == 1:
      # batch_size of 1
      row_index = None
    elif len(probs.shape) > 1:
      N = probs.shape[0]
      row_index = range(N)
    H = -np.mean(np.log(probs[row_index, target_index]))
    return H


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    if len(preds.shape) == 1:
      # batch_size of 1
      N = 1
      row_index = None
    elif len(preds.shape) > 1:
      N = preds.shape[0]
      row_index = range(N)
    d_preds = softmax(preds)
    loss = cross_entropy_loss(d_preds, target_index)
    d_preds[row_index, target_index] -= 1
    d_preds /= N
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.cache = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        X = self.cache
        d_result = np.multiply(d_out, X > 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        W, B = self.params()['W'].value, self.params()['B'].value
        out = np.dot(X, W) + B
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        W = self.params()['W'].value
        X = self.X
        
        # Gradient wrt to weights
        self.params()['W'].grad = np.dot(X.T, d_out)
        
        # Gradient wrt to bias
        self.params()['B'].grad = np.sum(d_out, axis=0).reshape(1,-1)
        
        # Gradient wrt to input
        d_input = np.dot(d_out, W.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
