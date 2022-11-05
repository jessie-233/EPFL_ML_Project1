# ReadMe
**Team members**: Rongchen Wang, Jin Yan, Jingqi Liu

**AIcrowd team**: Voyager_ML

## How to use: Launching `run.py`
In order to use `run.py`, follow these simple steps:
1. Put the *train.csv* and *test.csv* files in folder `Data`, as it is shown on github
2. Open a terminal and run `python run.py`
3. Wait for it to compute the model and create the submission file (it will take up to 10 min)
4. A new file *submission_result.csv* has been created under the same folder as `run.py`, which contains the prediction on testing set for AIcrowd competition.

## Code architecture
## 1. Root
The root directory contains final submisison files for Project 1
### `run.py`
The main script to load and preprocess the dataset, train a linear regression model, predict the labels of the test set, and create a submission file. Since we have already optimized the parameters in experiments, we use all training data as training set, without a validation set.

Parameters that can be changed here:
* the *degree* of the polynomial features (default = 11)

### `implementations.py`
Including ML functions used by `run.py`:
- Loss funcitons
- Computing gradients
- Optimizaiton methods (GD/SGD/Normal Equations)
- 6 ML regression models seen in the lecture. For `logistic_regression` (and `reg_logistic_regression`), the gradient descent algorithm is used, rather than stochastic gradient descent

Besides, it also contains functions for data processing:
- Loading the dataset
- Data pre-processing
- Split dataset into training and validating (only for experimenting)
- Creating a submission file

The pre-processing is composed of the following steps:
1. Load the dataset
2. Fill missing values with mean of each feature
3. Build the polynomial features
4. Build interactions
5. Z-score standardization
6. Add categorial features (one-hot coded)

Function `preprocessing` consolidates the above mentioned steps. It takes numeric and categorical features as input seperately, as well the degree of the polynomial features (indicated also with a flag `poly`). Interaction features will be added when flag `interaction` is True. Additionally, a flag `test` indicates if the input data is from test set. For training set, pre-processing should compute the mean and standard deviation of each feature and store them; and for test set, the ones provided will be used for standardization.


## 2. Experiments
This folder contains some experiments for Project 1 (not for submission)

### `implementations.py`
Including basic ML regression models

### `utilites.py`
Including data pre-processing and some other useful functions for experiments

### `data_explore.ipynb`
Some check-outs on dataset and data analytics, to get a general idea.

### `experiments.ipynb`
Record the results of different models and parameters used:
| Validation set accuracy & Runtime 	| Polynomials: NO<br>Interactions: NO<br>(Benchmark) 	| Polynomials: degree=10<br>Interactions: NO 	| Polynomials: degree=30<br>Interactions: NO 	| Polynomials: degree=9<br>Interactions: Yes 	| Polynomials: degree=10<br>Interactions: Yes 	| Polynomials: degree=11<br>Interactions: Yes 	| Polynomials: degree=12<br>Interactions: Yes 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Linear Regression 	| GD: 0.7454 (runtime: 2.4s)<br>NE: 0.7466 (runtime: 0.1s) 	| GD: 0.7712 (runtime: 29.3s)<br>NE: 0.8154 (runtime: 0.5s) 	| GD: 0.7047 (runtime: 1m18s)<br>NE: 0.8160 (runtime: 2.7s) 	| NE: 0.8221 	| NE: 0.8229 	| **NE: 0.8235** 	| NE: 0.8068 	|
| Logistic Regression 	| × 	| GD: 0.7791 (runtime: 2m6s)<br>SGD: 0.7850 (runtime: 59.7s) 	| × 	| × 	| × 	| × 	| × 	|

According to the experiments, when we add both polynomials (degree=11) and interactions, we got the best validation set accuracy 0.8235, which has 0.827 accuracy on testing set when evaluated on AIcrowd platform.  
