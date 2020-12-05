import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing

def rmsle(y_true,y_pred):
    '''
    Inputs:
    y_true: Actual y values
    y_pred: Predicted y values

    Output:
    Root-mean-squared-log error between y_true and y_pred
    '''
    return np.sqrt(mean_squared_log_error(y_true,y_pred))

def map_to_Gaussian(df,methodType):
    '''
    Mapping non-Gaussian distribution to Gaussian

    Input:
    df: DataFrame
    methodType: 'yeo-johnson' or 'box-cox'

    Output:
    Mapped dataframe
    '''
    
    pt=preprocessing.PowerTransformer(method=methodType,standardize=False)
    df=pt.fit_transform(df)

    return df

def remove_outlier(df,Serie):
    '''
    Removes outliers in one serie of a dataframe
    Input:
    df: DataFrame
    Serie: column name

    Output:
    Dataframe
    '''
    # removing above 3rd_quarter+1.5*IQR
    IQR=df[Serie].quantile(q=0.75)-df[Serie].quantile(q=0.25)
    df=df[df[Serie]<=(df[Serie].quantile(q=0.75)+1.5*IQR)]
    df.reset_index(inplace=True, drop=True)
    return df