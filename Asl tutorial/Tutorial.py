import os
import cv2
import numpy as np
from keras.models import load_model
import random
from tkinter import Tk, filedialog, Label, Button, messagebox

# Load the pre-trained model for sign language classification
model = load_model('ASL2.h5')

# Dictionary to map the predicted label index to the sign
label_mapping = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
    16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
    24: "Y", 25: "Z"
}

# Function to preprocess an image
def preprocess_image(image):
    # Resize and normalize the image
    resized_image = cv2.resize(image, (32, 32))
    normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Function to get a random alphabet from the model's predictions
def get_random_alphabet():
    alphabet_index = random.randint(0, 25)
    return label_mapping[alphabet_index]

# Function to ask the user to choose an image from the user_test directory
def choose_image_from_user_test():
    # Create Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the root window

    # Choose a file from the user_test directory
    file_path = filedialog.askopenfilename(initialdir=r"asl_alphabet\asl_user_test",
                                           title="Select an image",
                                           filetypes=[("Image files", "*.jpg *.png *.jpeg")])

    return file_path

# Function to ask for an alphabet and verify it
def ask_and_verify_alphabet():
    # Get a random alphabet
    random_alphabet = get_random_alphabet()

    # Display the chosen alphabet
    messagebox.showinfo("Chosen Alphabet", f"Please select an image of the alphabet: {random_alphabet}")

    # Choose an image from user_test directory
    image_path = choose_image_from_user_test()

    if not image_path:
        messagebox.showerror("Error", "No image selected.")
        return

    # Read the chosen image
    chosen_image = cv2.imread(image_path)

    # Preprocess the chosen image
    preprocessed_chosen_image = preprocess_image(chosen_image)

    # Make a prediction using the pre-trained model
    predictions = model.predict(preprocessed_chosen_image)
    predicted_label_index = np.argmax(predictions)
    predicted_alphabet = label_mapping[predicted_label_index]

    # Verify if the prediction matches the random alphabet
    if predicted_alphabet == random_alphabet:
        messagebox.showinfo("Result", "Correct!")
    else:
        messagebox.showinfo("Result", "Incorrect.")

# Ask and verify the alphabet
ask_and_verify_alphabet()
