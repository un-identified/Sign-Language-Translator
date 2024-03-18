import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow SavedModel
model = tf.saved_model.load("ASL\ASL_words_trained\model.savedmodel")

# Function to preprocess input frames
def preprocess_frame(frame):
    # Resize the frame to match the input size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values to the range [0, 1]
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to predict labels for a frame
# Dictionary to map index to label
label_mapping = {
    0: "Can",
    1: "hello",
    2: "Help",
    3: "Me",
    4: "Nobody",
    5: "Sad",
    6: "Understand",
    7: "Why"
}

# Function to predict labels for a frame
def predict_label(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Add batch dimension and convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(preprocessed_frame[np.newaxis, ...], dtype=tf.float32)
    # Perform inference
    predictions = model(input_tensor)
    # Convert predictions to labels
    label_index = np.argmax(predictions)
    label = label_mapping[label_index]  # Get the label corresponding to the index
    return label

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Predict label for the frame
    label = predict_label(frame)

    # Display the frame with the predicted label
    cv2.putText(frame, f"Predicted Label: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
