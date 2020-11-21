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
import matplotlib.pyplot as plt




# %% ----------------------------------------- 
# Data Ingestion
# --------------------------------------------
df=pd.read_csv('train.csv')



# Exploratory data analysis
# Assessing Y 
plt.figure()
plt.hist(df['SalePrice'])
plt.title('SalePrice with Outliers')
plt.show()

# Need to remove outliers in Y
# removing above 3rd_quarter+1.5*IQR
IQR=df['SalePrice'].quantile(q=0.75)-df['SalePrice'].quantile(q=0.25)
df=df[df['SalePrice']<=(df['SalePrice'].quantile(q=0.75)+1.5*IQR)]
df.reset_index(inplace=True, drop=True)

plt.figure()
plt.hist(df['SalePrice'],bins=40)
plt.title('SalePrice Outliers removed')
plt.show()


# %%

df.info()
df.shape[0]
df.head()
# %%
df.select_dtypes(exclude='object').columns.values.shape[0]
# %%
