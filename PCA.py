import numpy as np

from sklearn import decomposition #PCA Package
#from sklearn.decomposition import PCA #Alternative way
import pandas as pd

#3 features with 5 records
df1= pd.DataFrame({
        'SibSp':[1,2,3,40,5],
        'FamilySize':[2,4,7,12,10],
        'Age':[350,20,50,40,50],      
        'Fare':[100,200,300,400,4]})

# =============================================================================
# df1= pd.DataFrame({
#         'Age':[10,20,30,40,50],
#         'FamilySize':[2,4,6,8,10],
#         'SibSp':[1,2,3,4,5],        
#         'Fare':[100,200,300,400,500]}) #Age, FamilySize, Fare... Are features
# 
# =============================================================================
pca = decomposition.PCA(n_components=2) #n_components means, transform the data to n dimensions.

#find eigen values and eigen vectors of covariance matrix of df1
#.fit builds PCA model for given fetures to prinicpal components
#Equation: 
#PC1 = Age*w11+FamilySize*w12+Fare*w13.....
#PC2 = Age*w21+FamilySize*w22+Fare*w23.....
#PC3 = Age*w31+FamilySize*w32+Fare*w33.....
pca.fit(df1)
#print(pca.components_)
#convert all the data points from standard basis to eigen vector basis
df1_pca = pca.transform(df1)
print(df1_pca)

#variance of data along original axes
np.var(df1.Age) + np.var(df1.FamilySize) + np.var(df1.Fare)
#variance of data along principal component axes
#show eigen values of covariance matrix in decreasing order
pca.explained_variance_

np.sum(pca.explained_variance_)

#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#show the principal components
#show eigen vectors of covariance matrix of df
pca.components_[0]
pca.components_[1]
#pca.components_[2]


import os
import pandas as pd
from sklearn import decomposition #For PCA/Dimensionality Reduction
import seaborn as sns

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:/Users/prani/Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], axis=1)
titanic_train1.shape
X_train.info()

#Here comes the PCA!
pca = decomposition.PCA(n_components=2)
pca.fit(X_train)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#Transformation of PCA happens here
transformed_X_train = pca.transform(X_train)
transformed_X_train.shape

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

y_train = titanic_train['Survived']

#Assign transformed PCA data into new data frame for visualaiztion purpose
#transformed_df = pd.DataFrame(data = transformed_X_train, columns = ['pc1', 'pc2'])
#See whethere PC1 and PC2s are orthogonal are not!
#sns.jointplot('pc1', 'pc2', transformed_df)
