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
        self.params()['B'].grad = np.sum(d_out, axis=0).reshape(1, -1)

        # Gradient wrt to input
        d_input = np.dot(d_out, W.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # We need to save X to perform backward pass
        self.X = X
        pad_width, filter_size = self.padding, self.filter_size

        # Padding X:
        if self.padding:
            npad = ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0))
            X = np.pad(X, npad, 'constant', constant_values=0)
        stride = 1
        out_height = int((height - filter_size + 2*pad_width) / stride + 1)
        out_width = int((width - filter_size + 2*pad_width) / stride + 1)
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        conv_layer_out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_reshaped = self.W.value.reshape(-1, self.out_channels)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                Xf = X[:, y: y + filter_size, x: x + filter_size, :]
                out = (Xf.reshape(batch_size, -1)).dot(W_reshaped)
                conv_layer_out[:, y, x, :] = out + self.B.value
        return conv_layer_out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X = self.X
        pad_width, filter_size = self.padding, self.filter_size
        if self.padding:
            npad = ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0))
            X = np.pad(X, npad, 'constant', constant_values=0)

        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        Wgrad = np.zeros(self.W.value.shape)
        d_input = np.zeros(X.shape)
        W_reshaped = self.W.value.reshape(-1, self.out_channels)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                # Gradient wrt weights
                Xf = X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                out = Xf.reshape(batch_size, -1).T.dot(d_out[:, y, x, :])
                Wgrad += out.reshape(self.filter_size, self.filter_size, channels, out_channels)

                # Gradient wrt input
                gradinput = d_out[:, y, x, :].dot(W_reshaped.T).reshape(batch_size, self.filter_size, self.filter_size, channels)
                d_input[:, y: y + self.filter_size, x: x + self.filter_size, :] += gradinput

        # Cut off the padding
        if self.padding:
            d_input = d_input[:, pad_width:-pad_width, pad_width:-pad_width, :]

        # Gradient wrt to weights
        self.params()['W'].grad = Wgrad

        # Gradient wrt to bias
        self.params()['B'].grad = np.sum(d_out.reshape(-1, out_channels), axis=0)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        pool_size, stride = self.pool_size, self.stride
        out_height = int((height - pool_size)/stride+1)
        out_width = int((width - pool_size)/stride+1)
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        maxpoolout = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = {}
        for y in range(out_height):
            for x in range(out_width):
                X_F = X[:, y*stride: y*stride + pool_size, x*stride: x*stride + pool_size, :]
                maxpoolout[:, y, x, :] = np.amax(X_F, axis=(1, 2)).reshape(batch_size, channels)

                mask = np.zeros_like(X_F)
                X_F_rshpd = X_F.reshape(batch_size, pool_size*pool_size, channels)
                idx = X_F_rshpd.argmax(1)
                ax1, ax2 = np.indices((batch_size, channels))
                mask.reshape(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1
                self.mask[(y, x)] = mask
        return maxpoolout

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        pool_size, stride = self.pool_size, self.stride
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_input = np.zeros((batch_size, height, width, channels))
        for y in range(out_height):
            start_h = y*stride
            end_h = start_h + pool_size
            for x in range(out_width):
                start_w = x*stride
                end_w = start_w + pool_size
                mask = self.mask[(y, x)]
                d_input[:, start_h: end_h, start_w: end_w, :] += \
                    d_out[:, y, x, :].reshape(batch_size, 1, 1, channels)*mask
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
