
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="categorical",
    shuffle=False
)

class_names = test_data.class_names

# 2. Load the saved CNN model
model = tf.keras.models.load_model("fruit_quality_model.keras")

# 3. Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)

print("\n----- MODEL EVALUATION -----")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

# 4. Get actual labels
actual_labels = np.concatenate([
    labels.numpy() for images, labels in test_data
])

y_true = np.argmax(actual_labels, axis=1)

# 5. Predict test images
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

# 6. Display precision, recall and F1-score
print("\n----- CLASSIFICATION REPORT -----")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# 7. Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(xticks_rotation=45, cmap="Blues")
plt.title("Fruit Quality Classification - Confusion Matrix")
plt.tight_layout()

# 8. Save confusion matrix image
plt.savefig("confusion_matrix.png")

print("\nConfusion matrix saved successfully!")

plt.show()
