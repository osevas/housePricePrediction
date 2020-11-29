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
# removing above 3rd_quarter+1.5*IQR
IQR=df['SalePrice'].quantile(q=0.75)-df['SalePrice'].quantile(q=0.25)
df=df[df['SalePrice']<=(df['SalePrice'].quantile(q=0.75)+1.5*IQR)]
df.reset_index(inplace=True, drop=True)
# df is now main DataFrame in which outliers of Y have been removed.

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
# Feature Selection for continuous features (type=2)
# ----------------------------------------------
continuous_cols=column_df[column_df['col_category']==2]['cols'].to_list()
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
# Imputation for missing values
# --------------------------------------------
# using KNNImputer to fill in nulls in 
imputer=KNNImputer(n_neighbors=10,copy=False)
imputed=imputer.fit_transform(df_1[continuous_cols[:-1]],y=df_1['SalePrice'])
df_1[continuous_cols[:-1]]=imputed



#%%
#Trying Variance Threshold for feature selection
sel_variance=VarianceThreshold()
sel_variance_result=sel_variance.fit_transform(df_1[continuous_cols[:-1]])
print('Column count of Variance Threshold: {}'.format(sel_variance_result.shape[1]))
if sel_variance_result.shape[1]==df_1[continuous_cols[:-1]].shape[1]:
    print('Variance Threshold is not working\n')

#%% Trying f_regression for feature selection
f_test,p_val=f_regression(df_1[continuous_cols[:-1]],df_1['SalePrice'])
# print(f_test/f_test.max())

scores=f_test/f_test.max()

plt.figure()
plt.bar(x=np.arange(p_val.shape[0]),height=scores,label='F-scores normalized')
plt.xticks(ticks=np.arange(p_val.shape[0]),labels=continuous_cols[:-1],rotation='vertical')
plt.axhline(y=0.1)
plt.ylabel('F-test scores normalized')
plt.show()

df_1_f_reg=pd.DataFrame(data={'columns':continuous_cols[:-1],'F-scores normalized':scores})
plt.figure()
plt.hist(df_1_f_reg['F-scores normalized'])
plt.show()
# print(df_1_f_reg)

# DISCUSSION: set f-score limit as 0.1

f_score_lim=0.1

continousCols=df_1_f_reg[df_1_f_reg['F-scores normalized']>=f_score_lim]['columns'].to_list() # Column names that could be used in prediction
continousCols.append('SalePrice')

sns.pairplot(df_1[continousCols])
plt.show()

corr=np.corrcoef(df_1[continousCols].to_numpy().T)
sns.heatmap(corr)
plt.show()


#%% --------------------------------------------
# Prediction
# --------------------------------------------
df_2=df_1[continuousCols]
y=df_2['SalePrice']
X=df_2.drop(columns='SalePrice')

# to-do's
# 1) standardization
# 2) train_test_split
# 3) perform regression
