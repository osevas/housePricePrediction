import numpy as np
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true,y_pred):
    '''
    Inputs:
    y_true: Actual y values
    y_pred: Predicted y values

    Output:
    Root-mean-squared-log error between y_true and y_pred
    '''
    return np.sqrt(mean_squared_log_error(y_true,y_pred))