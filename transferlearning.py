import tensorflow as tf
import matplotlib.pyplot as plt

# LOAD DATASET
dataset = tf.keras.utils.image_dataset_from_directory(
    "DATASETS/cat vs dog/animals",
    shuffle=True,
    image_size=(224,224),
    batch_size=32
)

# CLASS NAMES
class_names = dataset.class_names
print("Classes :", class_names)

# SPLIT DATASET
dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)

train_dataset = dataset.take(train_size)
remaining = dataset.skip(train_size)
validation_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

# DISPLAY SAMPLE IMAGES
plt.figure(figsize=(8,8))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# LOAD PRETRAINED MODEL
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)


# FEATURE EXTRACTION USING FROZEN LAYERS
# Freeze all pretrained layers
base_model.trainable = False


# CREATE MODEL
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL
print("\nTraining Using Frozen Layers\n")
history_frozen = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)

# EVALUATE MODEL
loss1, accuracy1 = model.evaluate(test_dataset)
print("\nAccuracy Using Frozen Layers :", accuracy1)


# PARTIAL FINE TUNING
# Unfreeze model
base_model.trainable = True

# Freeze initial layers
# Train only last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# RECOMPILE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# TRAIN AGAIN
print("\nTraining Using Partial Fine Tuning\n")
history_finetune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=3
)

# EVALUATE AGAIN
loss2, accuracy2 = model.evaluate(test_dataset)
print("\nAccuracy After Fine Tuning :", accuracy2)

# COMPARISON
print("Frozen Layers Accuracy       :", accuracy1)
print("Fine Tuned Layers Accuracy   :", accuracy2)

methods = ["Frozen Layers", "Fine Tuning"]
accuracies = [accuracy1, accuracy2]
plt.figure(figsize=(6,5))
plt.bar(methods, accuracies)
plt.ylabel("Accuracy")
plt.title("Transfer Learning Performance Comparison")
plt.ylim(0,1)
for i in range(len(accuracies)):
    plt.text(i, accuracies[i] + 0.02,
             f"{accuracies[i]:.2f}",
             ha='center')
plt.show()