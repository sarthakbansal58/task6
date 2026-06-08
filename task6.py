# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot
import seaborn as sns


# %%
df=pd.read_csv("Iris.csv")
df.head()


# %%
df.info()

# %%
df["Species"].value_counts()
# here we have the categorical data we have to make the encoding of it

# %%
from sklearn.preprocessing import  LabelEncoder
lr=LabelEncoder()
df["Species"]=lr.fit_transform(df["Species"])

# %%
df.head()
# value is encoded

# %%
df["Species"].value_counts()
# Species
# 0    50   Iris-setosa 
# 1    50   Iris-versicolor
# 2    50   Iris-virginica 

 





# %%
# train the model to train the modle we fisrt have to split the data into train test split

from sklearn.model_selection import train_test_split

X=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=df["Species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
# now we have to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



# %%
# we have to tarin the model for the KNN
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# %%
knn.score(X_test,y_test)*100

# %%
knn.score(X_train,y_train)*100

# %%
y_pred = knn.predict(X_test)

print(y_pred)

# %%
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison)

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

# %%



