# -*- coding: utf-8 -*-
"""
useful functions including data pre-processing, cost funcions, and 6 ML functions in Table 1
"""

import numpy as np
import csv

##### data pre-processing, aggregated in preprocessing() #####
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]  # 22 is "PRI_jet_num" categorial column

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0  # set backgroud "b" to 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]  # take one every 50 records
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def fillna_with_mean(tx):
    """
    replace the missing value with mean value of each feature.
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features
    Returns:
        numpy array numpy array of shape (N,D), with missing values replaced with mean.
    """
    for feature in range(tx.shape[1]):
        row_indices = np.where(tx[:, feature] == -999.0)[0]
        clean_data = [x for x in tx[:, feature] if x != -999.0]
        tx[row_indices, feature] = np.mean(clean_data)
    return tx


def get_dummy(x, col=22):
    """
    one-hot encode the categorial feature in x. In this case, it's the 23rd column (index=22)
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
    Returns:
        dummies: numpy array of shape (N,d), one-hot encoded with d categories
    """
    target = x[:, col].astype(int)
    dummies = np.zeros((target.size, target.max() + 1))
    dummies[np.arange(target.size), target] = 1
    return dummies


def build_poly(x, degree):
    """polynomial basis functions for input data, for j = 1 up to j = degree.
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        degree: the largest power
    Returns:
        output: augmented data with polynomials
    """
    x = np.delete(x, 22, 1)  # firstly drop the categorical column 22
    output = np.copy(x)
    for i in range(1, degree):
        output = np.concatenate((output, np.power(x, i + 1)), axis=1)
    return output


def add_interactions(x):
    """create all interaction features of x
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
    Returns:
        output: augmented data with interation terms
    """
    x = np.delete(x, 22, 1)  # firstly drop the categorical column 22
    output = np.copy(x)
    for i in range(x.shape[1] - 1):
        for j in range(i + 1, x.shape[1]):
            combi = np.multiply(x[:, i], x[:, j])
            output = np.c_[output, combi]
    return output


def standardization(x, test=False, mean=None, std=None):
    """z-score standardization
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features
        test: a flag to decide whether standardization is performed on testing set.
              If True, mean and std must be given
    Returns:
        x: standardized data, mean at 0 and std at 1
        mean: mean of each column of training set, stored for future use on testing set
        std: std of each column of training set, stored for future use on testing set
    """
    # for training set:
    if not test:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    x = (x - mean) / std
    return x, mean, std


def make_x(x, dummies):
    """
    concatenate x, dummies, and a column of ones to make final x.
    Notice that dummy variables don't need to be standardized.
    """
    return np.concatenate((x, dummies, np.ones(len(x)).reshape((-1, 1))), axis=1)


def preprocessing(
    x, dummies, test=False, mean=None, std=None, poly=True, degree=10, interaction=False
):
    """
    A function to aggregate all data preprocessing steps.
    If poly=True, degree must be given
    If test=True, mean and std must be given
    """
    x = fillna_with_mean(x)
    if poly and interaction:
        x_poly = build_poly(x, degree=degree)
        x_interaction = add_interactions(x)
        x = np.c_[x_poly, x_interaction]
    elif poly and not interaction:
        x = build_poly(x, degree=degree)
    elif not poly and interaction:
        x = add_interactions(x)
    else:
        x = np.delete(x, 22, 1)

    x_stand, x_mean, x_std = standardization(x, test=test, mean=mean, std=std)
    final_x = make_x(x_stand, dummies)
    return final_x, x_mean, x_std


def split_data(x, y, ratio, seed=1):
    """
    split the dataset into training and validation set, based on the split ratio.
    Args:
        ratio: scalar in [0,1]
        seed: integer
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the validation data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the validation labels.
    """
    # set seed
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(x)))
    cut = int(np.floor(ratio * len(x)))
    x_tr = x[shuffle_indices[:cut]]
    x_te = x[shuffle_indices[cut:]]
    y_tr = y[shuffle_indices[:cut]]
    y_te = y[shuffle_indices[cut:]]
    return (x_tr, x_te, y_tr, y_te)


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


#### costs ####
def compute_loss(y, tx, w, method="MSE"):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D,). The vector of model parameters.
    """
    if method == "MSE":
        loss = 0.5 * np.square((y - tx @ w)).mean()  # 0.5 to be consistent with lecture
    elif method == "MAE":
        loss = np.abs((y - tx @ w)).mean()
    else:
        raise Exception("method {} not implemented!".format(method))
    return loss


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def compute_logistic_loss(y, tx, w):
    """Log-loss for logistic regression"""
    p = sigmoid(tx @ w)
    loss = -1 / len(y) * np.sum((1 - y) * np.log(1 - p) + y * np.log(p))
    return loss


##### gradient #####
def compute_gradient(y, tx, w, lambda_=0):
    """Computes the gradient at w."""
    return -1 / len(y) * (tx.T @ (y - tx @ w)) + 2 * lambda_ * w


def compute_logistic_gradient(y, tx, w, lambda_=0):
    """Gradient for logistic loss"""
    p = sigmoid(tx @ w)
    return -1 / len(y) * tx.T @ (y - p) + 2 * lambda_ * w


##### mini-batch #####
def batch_iter(y, tx, batch_size=1, num_batches=100, shuffle=True):
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
        if start_index < end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


##### (stochastic) gradient descent #####
def gradient_descent(
    y, tx, initial_w, max_iters=100, gamma=0.1, decay=0, lambda_=0, logistic=False
):
    """
    The Gradient Descent (GD) algorithm.
    Return the last recorded loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [
        compute_logistic_loss(y, tx, initial_w)
        if logistic
        else compute_loss(y, tx, initial_w)
    ]
    w = initial_w
    if logistic:
        for n_iter in range(max_iters):
            grad = compute_logistic_gradient(y, tx, w, lambda_=lambda_)
            w = w - gamma * grad
            loss = compute_logistic_loss(y, tx, w)
            gamma = (
                gamma * 1 / (1 + decay * n_iter)
            )  # add a Time-Based Learning Rate decay
            # store w and loss
            ws.append(w)
            losses.append(loss)
    else:
        for n_iter in range(max_iters):
            grad = compute_gradient(y, tx, w, lambda_=lambda_)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            gamma = (
                gamma * 1 / (1 + decay * n_iter)
            )  # add a Time-Based Learning Rate decay
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return (ws[-1], losses[-1])


def stochastic_gradient_descent(
    y, tx, initial_w, max_iters=100, gamma=0.1, decay=0, lambda_=0, logistic=False
):
    """
    The Stochastic Gradient Descent algorithm (SGD)
    Return the last recorded loss and weights
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [
        compute_logistic_loss(y, tx, initial_w)
        if logistic
        else compute_loss(y, tx, initial_w)
    ]
    w = initial_w
    # different method if is logistic reg
    if logistic:
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(
                y, tx, shuffle=True
            ):  # from from helpers import batch_iter
                grad = compute_logistic_gradient(
                    minibatch_y, minibatch_tx, w, lambda_=lambda_
                )
                w = w - gamma * grad
            loss = compute_logistic_loss(y, tx, w)
            gamma = (
                gamma * 1 / (1 + decay * n_iter)
            )  # add a Time-Based Learning Rate decay
            ws.append(w)
            losses.append(loss)
    else:
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(
                y, tx, shuffle=True
            ):  # from from helpers import batch_iter
                grad = compute_gradient(minibatch_y, minibatch_tx, w, lambda_=lambda_)
                w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            gamma = (
                gamma * 1 / (1 + decay * n_iter)
            )  # add a Time-Based Learning Rate decay
            ws.append(w)
            losses.append(loss)
    return (ws[-1], losses[-1])


########## 6 ML functions in Table 1 ##########
###############################################


def mean_squared_error_gd(
    y, tx, initial_w, max_iters=100, gamma=0.1, decay=0, lambda_=0
):
    """
    Linear regression using gradient descent
    """
    return gradient_descent(
        y,
        tx,
        initial_w,
        max_iters=max_iters,
        gamma=gamma,
        decay=decay,
        lambda_=lambda_,
        logistic=False,
    )


def mean_squared_error_sgd(
    y, tx, initial_w, max_iters=100, gamma=0.1, decay=0, lambda_=0
):
    """
    Linear regression using stochastic gradient descent (mini-batch-size 1)
    """
    return stochastic_gradient_descent(
        y,
        tx,
        initial_w,
        max_iters=max_iters,
        gamma=gamma,
        decay=decay,
        lambda_=lambda_,
        logistic=False,
    )


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
    try:
        w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    # for numerically unstable while not singular tx.T@tx
    except np.linalg.linalg.LinAlgError:
        w = np.linalg.lstsq(tx.T @ tx, tx.T @ y)[0]
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    """
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve((tx.T @ tx + 2 * lambda_ * N * np.identity(D)), tx.T @ y)
    mse = compute_loss(y, tx, w)
    return w, mse


def logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.1, decay=0):
    """
    Logistic regression using gradient descent
    """
    return gradient_descent(
        y,
        tx,
        initial_w,
        max_iters=max_iters,
        gamma=gamma,
        decay=decay,
        lambda_=0,
        logistic=True,
    )


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters=100, gamma=0.1, decay=0
):
    """
    Regularized logistic regression using gradient descent
    """
    return gradient_descent(
        y,
        tx,
        initial_w,
        max_iters=max_iters,
        gamma=gamma,
        decay=decay,
        lambda_=lambda_,
        logistic=True,
    )
