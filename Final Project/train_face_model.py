import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- PATHS ----------------
csv_path = r"C:\Users\Yathu\Desktop\mood_aware_project\dataset\fer2013.csv"
dataset_dir = r"C:\Users\Yathu\Desktop\mood_aware_project\dataset\face_emo"
train_dir = os.path.join(dataset_dir, "train")
test_dir  = os.path.join(dataset_dir, "test")
#-------make folder ------#
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ---------------- PARAMETERS ----------------
IMG_SIZE = 96
BATCH = 32
EPOCHS = 40
NUM_CLASSES = 7  # FER2013

emotion_labels = {
    0:"Angry",
    1:"Disgust",
    2:"Fear",
    3:"Happy",
    4:"Sad",
    5:"Surprise",
    6:"Neutral"
}

# ---------------- CSV TO IMAGES ----------------
def csv_to_images(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    for index, row in df.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48,48)

        emotion = emotion_labels[int(row['emotion'])]
        usage = row['Usage']  # 'Training' or 'PublicTest' or 'PrivateTest'

        # Determine folder
        folder = 'train' if 'Train' in usage else 'test'
        folder_path = os.path.join(save_dir, folder, emotion)
        os.makedirs(folder_path, exist_ok=True)

        img_name = f"{index}.jpg"
        cv2.imwrite(os.path.join(folder_path, img_name), image)

    print("CSV converted to images successfully!")

# Uncomment to convert CSV to images
# csv_to_images(csv_path, dataset_dir)

# ---------------- DATA GENERATORS ----------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2]
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH,
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH,
    shuffle=False
)

# ---------------- TRANSFER LEARNING MODEL ----------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

# ---------------- TRAIN ----------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ---------------- FINE-TUNING ----------------
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15
)

# ---------------- SAVE MODEL ----------------
model.save("best_face_emotion_model.h5")
print("Model saved successfully!")

# ---------------- PLOT ACCURACY & LOSS ----------------
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(8,6))
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# ---------------- CONFUSION MATRIX ----------------
pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)
