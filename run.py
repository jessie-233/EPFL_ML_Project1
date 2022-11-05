# -*- coding: utf-8 -*-
"""
Load data, train ML model, prediction on test dataset, and create standard submission for AIcrowd
"""

from implementations import *

#### some parameters ####
#########################

degree = 11

##### train a logistic regression model #####
#############################################

# prepare training data
y, x_train, _ = load_csv_data("./Data/train.csv", sub_sample=False)
dummies_train = get_dummy(x_train)
x_train, x_mean, x_std = preprocessing(
    x_train, dummies_train, test=False, poly=True, degree=degree, interaction=True
)


# train with linear regression using normal equations
model_weights, _ = least_squares(y, x_train)


##### Prediction on testing set #####
#####################################

# prepare test data
_, x_test, ids = load_csv_data("./Data/test.csv", sub_sample=False)
dummies_test = get_dummy(x_test)
x_test, _, _ = preprocessing(
    x_test,
    dummies_test,
    test=True,
    mean=x_mean,
    std=x_std,
    poly=True,
    degree=degree,
    interaction=True,
)
# predict with the trained model
y_pred = x_test @ model_weights
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = -1
# generate csv result
create_csv_submission(ids, y_pred, "submission_results.csv")
