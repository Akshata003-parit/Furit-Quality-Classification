
import tensorflow as tf

# Dataset folder paths
train_path = "dataset/train"
test_path = "dataset/test"

# Image settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# Load testing images
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Normalize pixel values from 0-255 to 0-1
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Display class names
print("Fruit Categories:", train_data.class_names if hasattr(train_data, "class_names") else "Loaded")

print("Training dataset loaded successfully!")
print("Testing dataset loaded successfully!")
