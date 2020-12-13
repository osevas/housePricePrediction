import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

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

def f_Regression(df,y,columnsInput):
    '''
    Selects columns with good correlation with target columns
    Input:
    df: DataFrame
    y: target column
    columnsInput: list of columns to evaluate

    Output:
    list of columns to select
    '''
    f_test,p_val=f_regression(df[columnsInput],y)
    scores=f_test/f_test.max()

    # DISCUSSION: set f-score limit
    f_score_lim=0.1

    plt.figure()
    plt.bar(x=np.arange(p_val.shape[0]),height=scores,label='F-scores normalized')
    plt.xticks(ticks=np.arange(p_val.shape[0]),labels=columnsInput,rotation='vertical')
    plt.axhline(y=f_score_lim)
    plt.ylabel('F-test scores normalized')
    plt.show()

    df_1_f_reg=pd.DataFrame(data={'columns':columnsInput,'F-scores normalized':scores})
    # plt.figure()
    # plt.hist(df_1_f_reg['F-scores normalized'])
    # plt.show()

    columnsToSelect=df_1_f_reg[df_1_f_reg['F-scores normalized']>=f_score_lim]['columns'].to_list() # Column names that could be used in prediction
    return columnsToSelect
    
