# -*- coding: utf-8 -*-
"""
useful functions including data pre-processing
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from implementations_ex import *

##### data pre-processing, aggregated in preprocessing() #####
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:] # 22 is "PRI_jet_num" categorial column

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0 # set backgroud "b" to 0

    # sub-sample
    if sub_sample:
        yb = yb[::50] # take one every 50 records
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def fillna_with_mean(tx):
    """
    replace the missing value with mean value of each feature.     
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        numpy array containing the data, with missing values replaced with mean.
    """
    for feature in range(tx.shape[1]):
        row_indices = np.where(tx[:,feature] == -999.0)[0]
        clean_data = [x for x in tx[:,feature] if x != -999.0]
        tx[row_indices,feature] = np.mean(clean_data)
    return tx


def get_dummy(x, col=22):
    """
    one-hot encode the categorial feature in x. In this case, it's the 23rd column
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        dummies: numpy array of shape (N,d), one-hot encoded with d categories
    """
    target = x[:, col].astype(int)
    dummies = np.zeros((target.size, target.max()+1))
    dummies[np.arange(target.size), target] = 1
    return dummies


def build_poly(x, degree):
    """polynomial basis functions for input data, for j = 0 up to j = degree.
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
        degree: the largest power
    Returns:
        output: numpy array containing the augmented data with polynomials (without 1)
    """
    x = np.delete(x, 22, 1) # firstly drop the categorical column 22
    output = np.copy(x)
    for i in range(1, degree):
        output = np.concatenate((output, np.power(x, i+1)), axis=1)
    return output


def add_interactions(x):
    """create all interaction features of x
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
    Returns:
        output: original x with added interation terms
    """
    x = np.delete(x, 22, 1) # firstly drop the categorical column 22
    output = np.copy(x)
    for i in range(x.shape[1]-1):
        for j in range(i+1, x.shape[1]):
            combi = np.multiply(x[:,i], x[:,j])
            output = np.c_[output, combi]
    return output

def standardization(x, test=False, mean=None, std=None):
    """z-score standardization; if poly is True, then reset the first column as 1
    Args:
        x: numpy array of shape (N,D), N is the number of samples, D is number of features        
        poly: whether the x is the result of build_poly()
    Returns:
        output: standardized data, mean at 1 and std at 1
        x_mean: mean of each column of training set, stored for future use on testing set
        x_std: std of each column of training set, stored for future use on testing set
    """
    # for training set:
    if not test:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    x = (x - mean) / std
    return x, mean, std


def make_x(x, dummies):
    """
    concatenate x and dummies by column, add a column with ones. 
    Notice that dummy variables don't need to be standardized.
    """
    return np.concatenate((x, dummies, np.ones(len(x)).reshape((-1,1))), axis=1)


def preprocessing(x, dummies, test=False, mean=None, std=None, poly=True, degree=10, interaction=False):
    """
    A function to aggregate all data preprocessing steps. 
    If poly=True, degree must be given
    If test=True, mean and std must be given
    """
    x = fillna_with_mean(x)
    if (poly and interaction):
        x_poly = build_poly(x, degree=degree)
        x_interaction = add_interactions(x)
        x = np.c_[x_poly, x_interaction]
    elif (poly and not interaction):
        x = build_poly(x, degree=degree)
    elif (not poly and interaction):
        x = add_interactions(x)
    else: 
        x = np.delete(x, 22, 1)
        
    x_stand, x_mean, x_std = standardization(x, test=test, mean=mean, std=std)
    final_x = make_x(x_stand, dummies)
    return final_x, x_mean, x_std


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing.     
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.  
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)
    shuffle_indices =  np.random.permutation(np.arange(len(x)))
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


######### plots #########

# plot loss
def plot_loss(train_log_loss, val_log_loss):
    plt.plot(train_log_loss, label="train")
    plt.plot(val_log_loss, label="validation")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()

##### accuracy #####
# calculate accuracy
def predict_acc(x_val, y_val, best_weights, logistic=False):
    if logistic:
        y_pred = sigmoid(x_val @ best_weights)
    else:
        y_pred = x_val @ best_weights
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    accuracy = (y_pred == y_val).sum() / len(y_val)
    print("The Accuracy is: %.4f"%accuracy)
