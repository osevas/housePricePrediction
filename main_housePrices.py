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
# Imputation for missing values
# --------------------------------------------


#%% --------------------------------------------
# Feature Selection for continuous features
# ----------------------------------------------
continuous_cols=column_df[column_df['col_category']==2]['cols'].to_list()
df_1_contin=df_1[continuous_cols[:-1]] #removing Y column
df_1_contin[df_1_contin['LotFrontage'].isnull()==True] # column 'LotFrontage' has null values

#Trying Variance Threshold for feature selection
sel_variance=VarianceThreshold()
sel_variance_result=sel_variance.fit_transform(df_1_contin)
print('Column count of Variance Threshold: {}'.format(sel_variance_result.shape[1]))
if sel_variance_result.shape[1]==df_1_contin.shape[1]:
    print('Variance Threshold is not working\n')

# Trying f_regression for feature selection
f_test,p_val=f_regression(df_1_contin,df_1['SalePrice'])
print(f_test)

print(p_val)


# X_cols=['LotFrontage','LotArea','MasVnrArea'] #select a few columns for trials

# plt.figure()
# sns.pairplot(df_1[['LotFrontage','LotArea','SalePrice']])
# plt.show()
# # %%

# %%
