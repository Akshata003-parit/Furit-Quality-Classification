
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the trained CNN model
model = tf.keras.models.load_model("fruit_quality_model.keras")

# 2. Define the six fruit categories
class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

# 3. Load the new fruit image
image_path = "sample_fruit.jpg.jpeg"

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

# 4. Convert image into an array
image_array = tf.keras.utils.img_to_array(image)

# 5. Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# 6. Predict using CNN
predictions = model.predict(image_array)

# 7. Find the predicted category
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

# 8. Calculate confidence
confidence = predictions[0][predicted_index] * 100

# 9. Display prediction
print("\n----- FRUIT QUALITY PREDICTION -----")
print("Predicted Category:", predicted_class)
print("Confidence:", round(float(confidence), 2), "%")

# 10. Display the original image and result
plt.imshow(image)
plt.title(
    f"Prediction: {predicted_class}\n"
    f"Confidence: {confidence:.2f}%"
)
plt.axis("off")
plt.show()
