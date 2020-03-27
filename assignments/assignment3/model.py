import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        width, height, im_channels = input_shape

        out_height_conv1 = int((height - 3 + 2*1) / 1 + 1)
        out_width_conv1 = int((width - 3 + 2*1) / 1 + 1)

        out_height_maxpool1 = int((out_height_conv1 - 4)/2+1)
        out_width_maxpool1 = int((out_width_conv1 - 4)/2+1)

        out_height_conv2 = int((out_height_maxpool1 - 3 + 2*1) / 1 + 1)
        out_width_conv2 = int((out_width_maxpool1 - 3 + 2*1) / 1 + 1)

        out_height_maxpool2 = int((out_height_conv2 - 4)/2+1)
        out_width_maxpool2 = int((out_width_conv2 - 4)/2+1)

        neurons_in_fc = out_height_maxpool2*out_width_maxpool2*conv2_channels
        self.sequential = [
            ConvolutionalLayer(im_channels, conv1_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size = 4, stride = 2),
            ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size = 4, stride = 2),
            Flattener(),
            FullyConnectedLayer(neurons_in_fc, n_output_classes)
        ]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        params = self.params()
        for param_name, param in params.items():
            param.grad = np.zeros_like(param.grad)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        out = X
        for module in self.sequential:
            out = module.forward(out)
        loss, d_preds = softmax_with_cross_entropy(out, y)

        d_out = d_preds
        for module in self.sequential[::-1]:
            d_out = module.backward(d_out)
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = X
        for module in self.sequential:
            pred = module.forward(pred)
        pred = np.argmax(pred, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        for i, module in enumerate(self.sequential):
            for key, param in module.params().items():
                result[key + '_' + str(i)] = param
        return result
