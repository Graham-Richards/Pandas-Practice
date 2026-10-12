import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import Funcs as fn
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

df = pd.read_csv('Book.csv')

df_mice = pd.read_csv("ImputedData.csv")

df_knn = pd.read_csv("KNN_ImputedData.csv")

# 1. Graph a diagnosis distribution MICE

# select a column
diagnosis = df_imp["Diagnosis"]

# count each diagnosis
diagnosis_counts = diagnosis.value_counts()

# make the graph
ax = diagnosis_counts.plot(kind="barh")

# put most common diagnosis on the top
ax.invert_yaxis()

# Customize
plt.xlabel("Number of Patients")
plt.ylabel("Diagnosis")
plt.title("Diagnosis Distribution")

#save
plt.savefig("Diagnosis Distribution_MICE.png", bbox_inches="tight")

# show
plt.show()




# Compare PD CD34 distributions between KNN and MICE

column = "Peripheral Blood CD34+ Cells"

df_mice[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5, # makes semi transparent for overlap visualization
    label="MICE"
)

df_knn[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5, 
    label="KNN"
)

plt.xlabel("Peripheral Blood CD34+ Cells")
plt.ylabel("Frequency")
plt.title("PB CD34 Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()
plt.show()







# Compare platelet dist. between knn and mice

column = "Platelet"

df_mice[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="MICE"
)

df_knn[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="KNN"
)

plt.xlabel("Platelet")
plt.ylabel("Frequency")
plt.title("Platelet Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()

plt.savefig("Platelet Distribution.png", bbox_inches="tight")

plt.show()











# Compare Hgb dist. between knn and mice

column = "Eosinophil"

df_mice[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="MICE"
)

df_knn[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="KNN"
)

plt.xlabel("Platelet")
plt.ylabel("Frequency")
plt.title("Eosinophil Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()

plt.savefig("Eosinophil Dist.png", bbox_inches="tight")

plt.show()







# Compare collected cd34/kg dist. between knn and mice

column = "Collect Actual CD34 E06/kg"

df_mice[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="MICE"
)

df_knn[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="KNN"
)

plt.xlabel("Platelet")
plt.ylabel("Frequency")
plt.title("Collect Actual CD34 E06/kg Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()

plt.savefig("cd34 dist.png", bbox_inches="tight")

plt.show()








# compare efficiency to knn vs mice

column = "Efficiency_CD34kg_per_TBV"

df_mice[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="MICE"
)

df_knn[column].plot(
    kind="hist",
    bins=20,
    alpha=0.5,
    label="KNN"
)

plt.xlabel("Platelet")
plt.ylabel("Frequency")
plt.title("Efficiency_CD34kg_per_TBV Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()

plt.savefig("efficiency dist.png", bbox_inches="tight")

plt.show()










# making a kde chart

column = "Peripheral Blood CD34+ Cells"

# KDE for MICE
sns.kdeplot(
    data=df_mice,
    x=column,
    label="MICE",
    fill=True,
    alpha=0.3
)

# KDE for KNN
sns.kdeplot(
    data=df_knn,
    x=column,
    label="KNN",
    fill=True,
    alpha=0.3
)

# Customize
plt.xlabel("Peripheral Blood CD34+ Cells")
plt.ylabel("Density")
plt.title("PB CD34 Distribution: MICE vs KNN")
plt.legend()
plt.tight_layout()

# Save
plt.savefig(
    "PB_CD34_KDE_MICE_vs_KNN.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()









# making a kde chart

column = "Eosinophil"

# KDE for MICE
sns.kdeplot(
    data=df_mice,
    x=column,
    label="MICE",
    fill=True,
    alpha=0.3
)

# KDE for KNN
sns.kdeplot(
    data=df_knn,
    x=column,
    label="KNN",
    fill=True,
    alpha=0.3
)

# Customize
plt.xlabel("Eosinophil")
plt.ylabel("Density")
plt.title("Eosinophil KDE: MICE vs KNN")
plt.legend()
plt.tight_layout()

# Save
plt.savefig(
    "eosinophil kde.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()













# make a scatterplot

x = "BMI"
y = "Peripheral Blood CD34+ Cells"

# MICE
sns.scatterplot(
    data=df_mice,
    x=x,
    y=y,
    label="MICE",
    color="blue",
    alpha=0.6
)

# KNN
sns.scatterplot(
    data=df_knn,
    x=x,
    y=y,
    label="KNN",
    color="red",
    alpha=0.6
)

plt.xlabel("BMI")
plt.ylabel("Peripheral Blood CD34+ Cells")
plt.title("BMI vs PB CD34: MICE vs KNN")

plt.legend()
plt.tight_layout()
plt.savefig("BMI vs PB CD34.png")
plt.show()





















































