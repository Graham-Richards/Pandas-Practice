import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
import statistics
import funcs as fn

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer


df = pd.read_csv('C:/workspace/Book.csv')

# MICE Style Approach

# use fn to remove numerical values, and columns with >70% NaN
numdf = fn.structure_missing_values_numeric(df)

# get numeric columns
num_col = numdf.columns

# Create Iterative Imputer
imp = IterativeImputer(max_iter=10, random_state=0)

# Impute missing numeric values
numdf[num_col] = imp.fit_transform(numdf[num_col])

#use fn to remove categorical values, and columns with >70% NaN
catdf = fn.structure_missing_values_categorical(df)

# get cat. columns
cat_col = catdf.select_dtypes(include="object").columns

# Create Simple Imputer
imp = SimpleImputer(strategy='most_frequent')

# Impute missing categorical values
catdf[cat_col] = imp.fit_transform(catdf[cat_col])

# COMBINE NUMERIC + CATEGORICAL
df_imputed = pd.concat([numdf, catdf], axis=1)

# Save
df_imputed.to_csv("ImputedData.csv", index=False)

display(df_imputed)




# KNN Approach

# Get usable numeric data
numdf = fn.structure_missing_values_numeric(df)

# Get numeric columns
num_col = numdf.columns

# Create KNN Imputer
knn_imp = KNNImputer(n_neighbors=5)

# Impute missing numeric values
numdf[num_col] = knn_imp.fit_transform(numdf[num_col])

# Get usable categorical data
catdf = fn.structure_missing_values_categorical(df)

# Get categorical columns
cat_col = catdf.select_dtypes(include="object").columns

# Create categorical imputer
cat_imp = SimpleImputer(strategy="most_frequent")

# Impute missing categorical values
catdf[cat_col] = cat_imp.fit_transform(catdf[cat_col])

# combine both df
df_knn = pd.concat([numdf, catdf], axis=1)

# Save
df_knn.to_csv("KNN_ImputedData.csv", index=False)

display(df_knn)

