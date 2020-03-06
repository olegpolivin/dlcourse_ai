import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.sequential = [
          FullyConnectedLayer(n_input, hidden_layer_size),
          ReLULayer(),
          FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
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
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_reg = 0
        params = self.params()
        for param_name, param in params.items():
          param.grad += 2*self.reg*param.value
          loss_reg += np.sum(param.value**2)
        return loss + self.reg * loss_reg

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = X
        for module in self.sequential:
          pred = module.forward(pred)
        pred = np.argmax(pred, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for i, module in enumerate(self.sequential):
          for key, param in module.params().items():
            result[key + '_' + str(i)] = param
        return result
