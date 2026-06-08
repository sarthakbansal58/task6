# K-Nearest Neighbors (KNN) Classification on Iris Dataset

## Objective

The objective of this project is to implement the **K-Nearest Neighbors (KNN)** classification algorithm using the **Iris Dataset** and evaluate its performance using accuracy and confusion matrix.

---

# Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VS Code

---

# Dataset

The project uses the **Iris Dataset**, which contains information about three species of Iris flowers.

### Features

* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm

### Target Variable

* Species

After label encoding:

* 0 → Iris-setosa
* 1 → Iris-versicolor
* 2 → Iris-virginica

---

# Project Workflow

## 1. Import Libraries

The required Python libraries are imported for data manipulation, visualization, preprocessing, model training, and evaluation.

---

## 2. Load Dataset

The Iris dataset is loaded using Pandas.

```python
df = pd.read_csv("Iris.csv")
```

---

## 3. Explore Dataset

The dataset structure and data types are checked using:

```python
df.head()

df.info()

df["Species"].value_counts()
```

---

## 4. Label Encoding

Since the **Species** column contains categorical values, it is converted into numerical values using **LabelEncoder**.

```python
from sklearn.preprocessing import LabelEncoder

lr = LabelEncoder()

df["Species"] = lr.fit_transform(df["Species"])
```

Encoded values:

| Species         | Encoded Value |
| --------------- | ------------- |
| Iris-setosa     | 0             |
| Iris-versicolor | 1             |
| Iris-virginica  | 2             |

---

## 5. Feature Selection

Input Features:

* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm

Target:

* Species

```python
X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

y = df["Species"]
```

---

## 6. Train-Test Split

The dataset is divided into training and testing sets.

* Training Data = 80%
* Testing Data = 20%

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## 7. Feature Scaling

The data is normalized using **StandardScaler** because KNN is a distance-based algorithm.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

---

## 8. KNN Model Training

The KNN classifier is created with **K = 1** and trained using the training dataset.

```python
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
```

---

## 9. Model Accuracy

Training Accuracy:

```python
knn.score(X_train, y_train) * 100
```

Testing Accuracy:

```python
knn.score(X_test, y_test) * 100
```

The model accuracy is calculated on both training and testing datasets.

---

## 10. Prediction

Predictions are generated for the test dataset.

```python
y_pred = knn.predict(X_test)
```

---

## 11. Actual vs Predicted Comparison

A comparison table is created to compare the actual and predicted values.

```python
comparison = pd.DataFrame({

"Actual": y_test.values,

"Predicted": y_pred

})
```

This helps verify the classification results.

---

## 12. Confusion Matrix

The confusion matrix is used to evaluate the classification performance.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

It shows the number of correct and incorrect predictions for each class.

---

# Advantages of KNN

* Simple and easy to understand
* No explicit training phase
* Works well for small datasets
* Supports multi-class classification

---

# Disadvantages of KNN

* Slow prediction for large datasets
* Sensitive to feature scaling
* Sensitive to noisy data
* Choosing the correct value of K is important

---

# Learning Outcomes

* Data preprocessing
* Label Encoding
* Train-Test Split
* Feature Scaling
* K-Nearest Neighbors Classification
* Model Prediction
* Accuracy Evaluation
* Confusion Matrix Analysis

---

# Conclusion

This project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm on the Iris dataset. The data is preprocessed using label encoding and feature scaling before training the model. The trained model predicts flower species with high accuracy, and its performance is evaluated using accuracy scores and a confusion matrix, making it a simple and effective example of supervised machine learning classification.
