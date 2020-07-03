import matplotlib.pyplot as plt
import numpy as np

from Dimensions import *

parameters = {}
activations = {}


def calculate_layer_dims(feature_length, output_length):
    return [feature_length] + LAYERS + [output_length]


def initialize_pararmeters(layer_dims):
    np.random.seed(1)
    for layer in range(1, len(layer_dims)):
        parameters["W" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) * 0.01
        parameters["b" + str(layer)] = np.zeros((layer_dims[layer], 1))


def sigmoid_activation(value):
    return 1 / (1 + np.exp(-value))


def relu_activation(value):
    return np.maximum(0, value)


def tanh_activation(value):
    return np.tanh(value)


def cross_entropy_loss(y_hat, y):
    return np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat + 1e-9)))


def calculate_loss(y_hat, y):
    m = y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = -1 / m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    ### END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())
    return cost


# def forward_propagation(parameters, training_set_x, layer_dims):
#     a_current = training_set_x
#     for layer in range(1, len(layer_dims) - 1):
#         w, b = parameters["W" + str(layer)], parameters["b" + str(layer)]
#         a_current = relu_activation(hypothesis(w, b, a_current))
#
#     # for last output node
#     w, b = parameters["W" + str(layer)], parameters["W" + str(layer)]
#     a_current = sigmoid_activation(hypothesis(w, b, a_current))
#
#     return a_current


def hypothesis(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z = hypothesis(A_prev, W, b)
        A = sigmoid_activation(Z)
    elif activation == "relu":
        Z = hypothesis(A_prev, W, b)
        A = relu_activation(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = ((A_prev, W, b), Z)
    return A, cache


def L_model_forward(X):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL+1e-9))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(grads, learning_rate, layer_dims):
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]


def L_layer_model(X, Y, learning_rate=LEARNING_RATE, num_iterations=ITERATIONS, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    layers_dims = calculate_layer_dims(X.shape[0], Y.shape[0])  # y.shape[0] is not proper way to get output labels
    np.random.seed(1)
    costs = []
    print (layers_dims)
    initialize_pararmeters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X)
        # print (AL)
        cost = calculate_loss(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        # print (grads)
        update_parameters(grads, learning_rate, layers_dims)
        # print (parameters)
        if print_cost:
            print("Cost after iteration %i: " % (i), cost)
        if print_cost and i % 50 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def predict(X, y):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    predictions -- predictions for the given dataset X
    """

    m = X.shape[1]
    # Forward propagation
    probas, caches = L_model_forward(X)
    predictions = probas > 0.5
    print("Accuracy: " + str(np.sum((predictions == y) / m)))

    return predictions


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])] + " \n Class: " + classes[y[0, index]])
