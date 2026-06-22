import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# Dataset Paths
file1 = "dataset/emotion_dataset.csv"  
file2 = "dataset/emo.csv"            


# Check if files exist
if not os.path.exists(file1):
    raise FileNotFoundError(f"{file1} not found! Please check your dataset folder.")
if not os.path.exists(file2):
    raise FileNotFoundError(f"{file2} not found! Please check your dataset folder.")

# Load Datasets Safely

# File1: no header, assign names
d1 = pd.read_csv(file1, header=None, quotechar='"', on_bad_lines='skip')
d1.columns = ['text','emotion']

# File2: has header, keep only necessary columns
d2 = pd.read_csv(file2, quotechar='"', on_bad_lines='skip')
d2 = d2[['text','emotion']]

# Merge datasets
data = pd.concat([d1, d2], ignore_index=True)

# Remove classes with <2 samples

counts = data['emotion'].value_counts()
valid_classes = counts[counts > 1].index
data = data[data['emotion'].isin(valid_classes)]

print("Class distribution after removing rare classes:")
print(data['emotion'].value_counts())


# Prepare Data

X = data['text']
y = data['emotion']

# Text vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.25, random_state=42, stratify=y
)


# Train Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions & Evaluation


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n",
       classification_report(y_test, y_pred))


# Save Model & Vectorizer

pickle.dump(model, open("text_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\nModel and vectorizer saved successfully.")


# Confusion Matrix Plot 

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Text Sentiment Analysis - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# Learning Curve (Training vs Validation Accuracy)

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_vec,
    y,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, val_mean, label="Validation Accuracy")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Text Sentiment Analysis - Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curve.png")
plt.show()
print("Learning curve saved as learning_curve.png")
