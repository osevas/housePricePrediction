# Subject: Predicting house prices
#   a Kaggle competition.  Data has been acquired from Kaggle.
# Author: Onur Sevket Aslan
# Date: 2020-11-21
# Evaluation: Submissions are evaluated on 
#   Root-Mean-Squared-Error (RMSE) between the 
#   logarithm of the predicted value and the logarithm 
#   of the observed sales price. (Taking logs means that 
#   errors in predicting expensive houses and cheap houses 
#   will affect the result equally.)

# %% ----------------------------------------- 
# Libraries
# --------------------------------------------
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_log_error
import functions
from sklearn.ensemble import AdaBoostRegressor





# %% ----------------------------------------- 
# Data Ingestion
# --------------------------------------------
df=pd.read_csv('train.csv')


# --------------------------------------------
# Exploratory data analysis
# --------------------------------------------

# Assessing Y 
fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].hist(df['SalePrice'])
axs[0].set_title('SalePrice with Outliers')

axs[1].boxplot(df['SalePrice'])
axs[1].set_title('SalePrice with Outliers')

plt.show()

# Need to remove outliers in Y

df=functions.remove_outlier(df,'SalePrice')

fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].hist(df['SalePrice'])
axs[0].set_title('SalePrice with no Outliers')

axs[1].boxplot(df['SalePrice'])
axs[1].set_title('SalePrice with no Outliers')

plt.show()


# %%--------------------------------------------
# Feature Analysis
# --------------------------------------------
# Categorizing features into types
#
# types of features:
# 1) categorical -> to be encoded
# 2) continuous -> with square feet unit
# 3) year
# 4) counts
# 5) price
# --------------------------------------------
# Columns will be assessed by evaluating columns separately

#Removing 'Id' column from the dataset
df_1=df.drop(columns='Id')
# df_1 is the main DataFrame in which 'Id' column has been removed.

# 1) Categorizing features
df_2=df_1.select_dtypes(exclude='object')
continuous_cols=df_2.columns.values #columns names of df_1 that are in continuous type
column_df=pd.DataFrame(data={'cols':continuous_cols,'col_category':2*np.ones(df_2.columns.values.shape[0])},dtype='int')

# Specifying 'year' columns
column_df.iloc[5:7,1]=3
column_df.iloc[24,1]=3
column_df.iloc[35,1]=3

# Specifying 'price' columns
column_df.iloc[33,1]=5

# Specifying 'counts' columns
column_df.iloc[16:24,1]=4
column_df.iloc[25,1]=4
column_df.iloc[34,1]=4
# print(column_df)
# print('\n')

df_3=df_1.select_dtypes(include='object')
column1_df=pd.DataFrame(data={'cols':df_3.columns.values,'col_category':1*np.ones(df_3.columns.values.shape[0])},dtype='int')
column_df=column1_df.append(column_df)

# Delete unnecessary variable for clarity
del df_2,df_3,column1_df

#%% --------------------------------------------
# Feature Analysis for continuous features (type=2)
# ----------------------------------------------
continuous_cols=column_df[column_df['col_category']==2]['cols'].to_list()
continuous_cols=continuous_cols[:-1] #removing 'SalePrice' from continuous columns

# df_1_contin=df_1[continuous_cols] 
# df_1_contin[df_1_contin['LotFrontage'].isnull()==True] # column 'LotFrontage' and 'MasVnrArea' has null values

# Analyzing continuous columns with missing values
# Columns: 'LotFrontage' and 'MasVnrArea'
# Purpose: is KNNImputation appropriate for imputation

# df_1_contin.fillna(value=-10,inplace=True)

# fig, axs=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
# axs[0].scatter(df_1_contin['SalePrice'],df_1_contin['LotFrontage'])
# axs[0].set_title('LotFrontage')

# axs[1].scatter(df_1_contin['SalePrice'],df_1_contin['MasVnrArea'])
# axs[1].set_title('MasVnrArea')

# plt.show() # DISCUSSION: Missing values have data close to themselves.  KnnImputation will be appropriate

# ----------------------------------------------
# fig, axs=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
# axs[0].hist(df_1['LotFrontage'])
# axs[0].set_title('LotFrontage')

# axs[1].hist(df_1['MasVnrArea'])
# axs[1].set_title('MasVnrArea')

# plt.show()

# fig, axs=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
# axs[0].scatter(df_1['SalePrice'],df_1['LotFrontage'])
# axs[0].set_title('LotFrontage')

# axs[1].scatter(df_1['SalePrice'],df_1['MasVnrArea'])
# axs[1].set_title('MasVnrArea')

# plt.show()
# ----------------------------------------------

#%% --------------------------------------------
# Feature Analysis for columns with type=3
# ----------------------------------------------
year_cols=column_df[column_df['col_category']==3]['cols'].to_list()
# Column 'GarageYrBlt' has null values

fig, axs=plt.subplots(nrows=1,ncols=4,figsize=(20,20))
for i,col in enumerate(year_cols):
    axs[i].hist(df_1[col])
    axs[i].set_title(col)
plt.show()

fig, axs=plt.subplots(nrows=1,ncols=4,figsize=(20,20))
for i,col in enumerate(year_cols):
    axs[i].scatter(df_1[col],df_1['SalePrice'])
    axs[i].set_title(col)
plt.show()

#%% --------------------------------------------
# Feature Analysis for columns with type=4
# ----------------------------------------------
count_cols=column_df[column_df['col_category']==4]['cols'].to_list()
# count_cols have no null values

fig, axs=plt.subplots(nrows=1,ncols=10,figsize=(30,10))
for i,col in enumerate(count_cols):
    axs[i].hist(df_1[col])
    axs[i].set_title(col)
plt.show()

fig, axs=plt.subplots(nrows=1,ncols=10,figsize=(30,10))
for i,col in enumerate(count_cols):
    axs[i].scatter(df_1[col],df_1['SalePrice'])
    axs[i].set_title(col)
plt.show()

#%% --------------------------------------------
# Feature Analysis for columns with type=5
# ----------------------------------------------
price_cols=column_df[column_df['col_category']==5]['cols'].to_list()
# count_cols have no null values

plt.figure(figsize=(30,10))
plt.hist(df_1[price_cols[0]])
plt.title(price_cols[0])
plt.show()

plt.figure(figsize=(30,10))
plt.scatter(df_1[price_cols[0]],df_1['SalePrice'])
plt.title(price_cols[0])
plt.show()

#%% --------------------------------------------
# Imputation for missing values
# --------------------------------------------
# using KNNImputer to fill in nulls in 

continuous_cols=continuous_cols+year_cols+count_cols+price_cols #adding year columns to continuous columns

imputer=KNNImputer(n_neighbors=10,copy=False)
imputed=imputer.fit_transform(df_1[continuous_cols],y=df_1['SalePrice'])
df_1[continuous_cols]=imputed


#%% --------------------------------------------
# Feature Selection for continuous features (type=2)
# ----------------------------------------------
#Trying Variance Threshold for feature selection
sel_variance=VarianceThreshold()
sel_variance_result=sel_variance.fit_transform(df_1[continuous_cols])
print('Column count of Variance Threshold: {}'.format(sel_variance_result.shape[1]))
if sel_variance_result.shape[1]==df_1[continuous_cols].shape[1]:
    print('Variance Threshold is not working\n')

# Trying f_regression for feature selection
f_test,p_val=f_regression(df_1[continuous_cols],df_1['SalePrice'])
# print(f_test/f_test.max())

scores=f_test/f_test.max()

# DISCUSSION: set f-score limit
f_score_lim=0.1

plt.figure()
plt.bar(x=np.arange(p_val.shape[0]),height=scores,label='F-scores normalized')
plt.xticks(ticks=np.arange(p_val.shape[0]),labels=continuous_cols,rotation='vertical')
plt.axhline(y=f_score_lim)
plt.ylabel('F-test scores normalized')
plt.show()

df_1_f_reg=pd.DataFrame(data={'columns':continuous_cols,'F-scores normalized':scores})
# plt.figure()
# plt.hist(df_1_f_reg['F-scores normalized'])
# plt.show()

continousCols=df_1_f_reg[df_1_f_reg['F-scores normalized']>=f_score_lim]['columns'].to_list() # Column names that could be used in prediction
continousCols.append('SalePrice')
# sns.pairplot(df_1[continousCols])
# plt.show()

# corr=np.corrcoef(df_1[continousCols].to_numpy().T)
# sns.heatmap(corr)
# plt.show()

del continousCols[-1]

#%% --------------------------------------------
# Outlier removal from features
# ----------------------------------------------
# Removing outliers in selected columns that need outlier removal
# First i analyzed the features separately and then decided which column needs outlier removal
for column in continousCols[1:5]:
    df_1=functions.remove_outlier(df_1,column)

#%% --------------------------------------------
# Train-valid-split & Scaling
# --------------------------------------------
y=df_1['SalePrice']
X=df_1[continousCols]

# train_valid_split
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.40)

# Tried mapping non-Gaussian features to Gaussian, but it did not give good result
#X_train[['MasVnrArea','BsmtFinSF1']]=functions.map_to_Gaussian(X_train[['MasVnrArea','BsmtFinSF1']],'yeo-johnson')

#%%
# Scaling
# scaler=preprocessing.StandardScaler() # StandardScaler produces negative values, which become problematic later in calculating RMLSE
# X_train_scaled=scaler.fit_transform(X_train)

X_train_scaled=preprocessing.minmax_scale(X_train,feature_range=(0,1),axis=0,copy=True)

# X_train_scaled=pd.DataFrame(data=X_train_scaled,columns=continousCols)
# # Evaluating scaling of train
# fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,20))
# for i in range(5):
#     ax[i,0].hist(X_train.iloc[:,i])
#     ax[i,1].hist(X_train_scaled[:,i])
#     ax[i,0].set_title(continousCols[i])

# plt.show()




# Scaling validation set
# X_valid_scaled=scaler.transform(X_valid) # StandardScaler produces negative values, which become problematic later in calculating RMLSE

X_valid_scaled=preprocessing.minmax_scale(X_valid,feature_range=(0,1),axis=0,copy=True)

# X_valid_scaled=pd.DataFrame(data=X_valid_scaled,columns=continousCols)
# # Evaluating scaling of validation
# fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(10,20))
# for i in range(5):
#     ax[i,0].hist(X_valid.iloc[:,i])
#     ax[i,1].hist(X_valid_scaled[:,i])
#     ax[i,0].set_title(continousCols[i])

# plt.show()

# convert 2 variables to Gaussion
# MasVnrArea, BsmtFinSF1

#%% --------------------------------------------
# Fit & Prediction of Linear Regression
# --------------------------------------------

# Models to try:
# 1) Ordinary least squares
# 2) Decision trees
# 3) Ensemble method -> Random Forest

# Ordinary least squares
reg=linear_model.LinearRegression()
reg.fit(X_train_scaled,y_train)

train_r2_linreg=reg.score(X_train_scaled,y_train)
test_r2_linreg=reg.score(X_valid_scaled,y_valid)

print('Train R^2 of ordinary least squares: {:.3}'.format(train_r2_linreg)) #calculating train accuracy
print('Test R^2 of ordinary least squares: {:.3}'.format(test_r2_linreg)) #calculating valid accuracy

#Linear Regression predicts a negative value.  This causes RMSLE not to be calculated.

# y_train_predict=reg.predict(X_train_scaled)
# y_valid_predict=reg.predict(X_valid_scaled)

# train_rmsle_linreg=functions.rmsle(y_train,y_train_predict)
# test_rmsle_linreg=functions.rmsle(y_valid,y_valid_predict)

# print('Train RMSLE of ordinary least squares: {:.3}'.format(train_rmsle_linreg)) #calculating train rmsle
# print('Test RMSLE of ordinary least squares: {:.3}'.format(test_rmsle_linreg)) #calculating valid rmsle

#%% --------------------------------------------
# Fit & Prediction of Decision Tree Regressor
# --------------------------------------------
clf=tree.DecisionTreeRegressor(max_depth=4)
clf=clf.fit(X_train_scaled,y_train)

y_train_predict=clf.predict(X_train_scaled)
y_valid_predict=clf.predict(X_valid_scaled)

train_r2_DTreg=clf.score(X_train_scaled,y_train)
test_r2_DTreg=clf.score(X_valid_scaled,y_valid)

train_rmsle_DTreg=functions.rmsle(y_train,y_train_predict)
test_rmsle_DTreg=functions.rmsle(y_valid,y_valid_predict)

print('Evaluation of DecisionTreeRegressor:\n')
print('Train R^2 of DecisionTreeRegressor: {:.3}'.format(train_r2_DTreg)) #calculating train accuracy
print('Test R^2 of DecisionTreeRegressor: {:.3}'.format(test_r2_DTreg)) #calculating valid accuracy
print('Train RMSLE of DecisionTreeRegressor: {:.3}'.format(train_rmsle_DTreg)) #calculating train rmsle
print('Test RMSLE of DecisionTreeRegressor: {:.3}'.format(test_rmsle_DTreg)) #calculating valid rmsle

#%% --------------------------------------------
# Fit & Prediction of AdaBoost Regressor
# --------------------------------------------
regr=AdaBoostRegressor(loss='exponential')
regr.fit(X_train_scaled,y_train)

y_train_predict=regr.predict(X_train_scaled)
y_valid_predict=regr.predict(X_valid_scaled)

train_r2_ABreg=regr.score(X_train_scaled,y_train)
test_r2_ABreg=regr.score(X_valid_scaled,y_valid)

train_rmsle_ABreg=functions.rmsle(y_train,y_train_predict)
test_rmsle_ABreg=functions.rmsle(y_valid,y_valid_predict)

print('Evaluation of AdaBoost Regressor:\n')
print('Train R^2 of AdaBoost Regressor: {:.3}'.format(train_r2_ABreg)) #calculating train accuracy
print('Test R^2 of AdaBoost Regressor: {:.3}'.format(test_r2_ABreg)) #calculating valid accuracy
print('Train RMSLE of AdaBoost Regressor: {:.3}'.format(train_rmsle_ABreg)) #calculating train rmsle
print('Test RMSLE of AdaBoost Regressor: {:.3}'.format(test_rmsle_ABreg)) #calculating valid rmsle


#%% --------------------------------------------
# Comparison of y_train and y_train_predict
# --------------------------------------------
j=4
fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
axs[0].scatter(X_train.iloc[:,j],y_train)
axs[0].scatter(X_train.iloc[:,j],y_train_predict,marker='x')
axs[0].set_title('y_train vs. y_train_predict')
axs[0].set_xlabel(X_train.columns.values[4])
axs[0].set_ylabel('SalePrice')

axs[1].scatter(X_valid.iloc[:,j],y_valid)
axs[1].scatter(X_valid.iloc[:,j],y_valid_predict,marker='x')
axs[1].set_title('y_valid vs. y_valid_predict')
axs[1].set_xlabel(X_train.columns.values[4])
axs[1].set_ylabel('SalePrice')
plt.show()
#%% --------------------------------------------
# Comparison of models
# --------------------------------------------
model_result=pd.DataFrame(data={'Model':['Linear Regression','Decision Tree Regressor','AdaBoost Regressor'],
    'Train R2':[train_r2_linreg,train_r2_DTreg,train_r2_ABreg],
    'Test R2':[test_r2_linreg,test_r2_DTreg,test_r2_ABreg],
    'Train RMSLE':['N/A',train_rmsle_DTreg,train_rmsle_ABreg],
    'Test RMSLE':['N/A',test_rmsle_DTreg,test_rmsle_ABreg]})
print(model_result)
#%% --------------------------------------------
# Residual plot
# --------------------------------------------
fig,axs=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
axs[0].hist(y_train-reg.predict(X_train_scaled))
axs[0].set_title('Residual plot of train set')

axs[1].hist(y_valid-reg.predict(X_valid_scaled))
axs[1].set_title('Residual plot of test set')
plt.show()

# %%
