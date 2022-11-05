# -*- coding: utf-8 -*-
"""
contains some necessary methods and 6 ML functions in Table 1
"""

import numpy as np

##### costs #####
def compute_loss(y, tx, w, method="MSE"):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D,). The vector of model parameters.
    """
    if method == "MSE":
        loss = 0.5 * np.square((y - tx @ w)).mean()
    elif method == "MAE": 
        loss = np.abs((y - tx @ w)).mean()
    else:
        raise Exception("method {} not implemented !".format(method)) 
    return loss


def sigmoid(x):
	""" sigmoid function """
	return 1 / (1 + np.exp(-x))


def compute_logistic_loss(y, tx, w):
    """ Log-loss for logistic regression """
    p = sigmoid(tx @ w)
    loss = -1 / len(y) * np.sum((1 - y) * np.log(1 - p) + y * np.log(p))
    return loss


##### gradient #####
def compute_gradient(y, tx, w, lambda_=0):
    """Computes the gradient at w.
    """
    return - 1/ len(y) * (tx.T @ (y - tx @ w)) + 2* lambda_*w


def compute_logistic_gradient(y, tx, w, lambda_=0):
    """ Gradient for logistic loss """
    p = sigmoid(tx @ w)
    return -1/len(y) * tx.T@(y-p) + 2* lambda_*w


##### mini-batch #####
def batch_iter(y, tx, batch_size=1, num_batches=10000, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    batch_size: must be 1 according to task description
    num_batches: can be changed freely
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


##### (stochastic) gradient descent #####
def gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0, logistic=False):
    """The Gradient Descent (GD) algorithm.
    Return the last recorded loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    train_losses = []
    val_losses = []
    w = initial_w
    if logistic:
        for n_iter in range(max_iters):
            grad = compute_logistic_gradient(y, tx, w, lambda_=lambda_)
            w = w - gamma * grad
            loss_tr = compute_logistic_loss(y, tx, w)
            loss_va = compute_logistic_loss(y_va, x_va, w)
            gamma = gamma * 1 / (1 + decay * n_iter) # add a Time-Based Learning Rate decay
            # store w and loss
            ws.append(w)
            train_losses.append(loss_tr)
            val_losses.append(loss_va)
    else:
        for n_iter in range(max_iters):
            grad = compute_gradient(y, tx, w, lambda_=lambda_)
            w = w - gamma * grad
            loss_tr = compute_loss(y, tx, w)
            loss_va = compute_loss(y_va, x_va, w)
            gamma = gamma * 1 / (1 + decay * n_iter) # add a Time-Based Learning Rate decay
            # store w and loss
            ws.append(w)
            train_losses.append(loss_tr)
            val_losses.append(loss_va)
    best_iter = np.argmin(np.array(val_losses)) - 1
    print("The {bi} of the {ti} GD iters has the lowest val loss: loss(train)={l1}, loss(val)={l2}".format(bi=best_iter+2, ti=max_iters, l1=train_losses[best_iter], l2=val_losses[best_iter]))
    return (ws[best_iter], train_losses, val_losses) 


def stochastic_gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0, logistic=False):
    """The Stochastic Gradient Descent algorithm (SGD)
    Return the last recorded loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    train_losses = []
    val_losses = []
    w = initial_w
    # different method if is logistic reg
    if logistic:
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, shuffle=True): # from from helpers import batch_iter
                grad = compute_logistic_gradient(minibatch_y, minibatch_tx, w, lambda_=lambda_)
                w = w - gamma * grad
            loss_tr = compute_logistic_loss(y, tx, w)
            loss_va = compute_logistic_loss(y_va, x_va, w)
            gamma = gamma * 1 / (1 + decay * n_iter) # add a Time-Based Learning Rate decay
            ws.append(w)
            train_losses.append(loss_tr)
            val_losses.append(loss_va)
    else:
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(y, tx, shuffle=True): # from from helpers import batch_iter
                grad = compute_gradient(minibatch_y, minibatch_tx, w, lambda_=lambda_)
                w = w - gamma * grad
            loss_tr = compute_loss(y, tx, w)
            loss_va = compute_loss(y_va, x_va, w)
            gamma = gamma * 1 / (1 + decay * n_iter) # add a Time-Based Learning Rate decay
            ws.append(w)
            train_losses.append(loss_tr)
            val_losses.append(loss_va)
    best_iter = np.argmin(np.array(val_losses)) - 1
    print("The {bi} of the {ti} GD iters has the lowest val loss: loss(train)={l1}, loss(val)={l2}".format(bi=best_iter+2, ti=max_iters, l1=train_losses[best_iter], l2=val_losses[best_iter]))
    return (ws[best_iter], train_losses, val_losses) 


########## 6 ML functions in Table 1 ##########
###############################################

# Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0):
    return gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=max_iters, gamma=gamma, decay=decay, lambda_=lambda_, logistic=False)


# Linear regression using stochastic gradient descent (mini-batch-size 1)
def mean_squared_error_sgd(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0):
    return stochastic_gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=max_iters, gamma=gamma, decay=decay, lambda_=lambda_, logistic=False)


# Least squares regression using normal equations
def least_squares(y, tx):
    try:
        w = np.linalg.solve(tx.T@tx, tx.T@y)
    # for numerically unstable while not singular tx.T@tx
    except np.linalg.linalg.LinAlgError:
        w = np.linalg.lstsq(tx.T@tx, tx.T@y)[0]
    mse = compute_loss(y, tx, w)
    return w, mse


# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve((tx.T @ tx + 2*lambda_ *N*np.identity(D)), tx.T@y)
    mse = compute_loss(y, tx, w) 
    return w, mse


# Logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0):
    return gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=max_iters, gamma=gamma, decay=decay, lambda_=0, logistic=True)


# Regularized logistic regression using gradient descent
def reg_logistic_regression(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0.1):
    return gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=max_iters, gamma=gamma, decay=decay, lambda_=lambda_, logistic=True)

# Regularized logistic regression using stochastic gradient descent
def reg_logistic_regression_sgd(y, tx, initial_w, y_va, x_va, max_iters=100, gamma=0.1, decay=0, lambda_=0.1):
    return stochastic_gradient_descent(y, tx, initial_w, y_va, x_va, max_iters=max_iters, gamma=gamma, decay=decay, lambda_=lambda_, logistic=True)
