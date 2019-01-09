import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Load the training dataset and test dataset into dataframes train_set and test_set respectively.

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

#Exploring Data
print('\n')
print(train_set.SalePrice.describe(),'\n')
#sns.distplot(train_set.SalePrice,kde=False)
"""
Negative skewing is observed. Hence, log of saleprice is taken to try and control skewing.
"""
target = np.log(train_set.SalePrice)
#sns.distplot(target,kde=False)
numeric_features = train_set.select_dtypes(include=[np.number])
categoricals = train_set.select_dtypes(exclude=[np.number])
#print(numeric_features.dtypes,'\n')
correlation = numeric_features.corr()
"""
On checking the correlation between numerical features with our target value,
it's observed that highly correlated features are OverallQual, GrLivArea, GarageCars, GarageArea
and poorly correlated features are YrSold, OverallCond, MSSubClass, EnclosedPorch, KitchenAbvGr.
"""
print(correlation['SalePrice'].sort_values(ascending=False),'\n')
#sns.barplot('OverallQual','SalePrice',data=train_set)  #Highly correlated
#sns.scatterplot('GrLivArea',target,data=train_set)    #Highly correlated
#sns.scatterplot('GarageArea',target,data=train_set)   #Highly correlated
#sns.scatterplot('1stFlrSF',target,data=train_set)    #Highly correlated
#sns.scatterplot('KitchenAbvGr','SalePrice',data=train_set) #No correlation

#Description of categorical features
#print(categoricals.count(),'\n')
print(categoricals.describe(),'\n')
print(categoricals.count(),'\n')
#print(train_set.BsmtExposure.value_counts())
#sns.barplot('BsmtExposure','SalePrice',data=train_set)
"""
Get an idea of which features hae the most Null values
"""
#sns.heatmap(train_set.isnull())

plt.show()