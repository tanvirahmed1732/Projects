import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import numpy as np
import joblib

# Load the trained model and PCA transformer
model = joblib.load('digit_recognition_model.pkl')
pca = joblib.load('pca_transformer.pkl')

# Function to process the image (resize to 250x350, convert to grayscale, and flatten)
# Function to process the image (resize to 250x350, convert to grayscale, and flatten)
def process_image(image_path):
    # Open the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((250, 350), Image.Resampling.LANCZOS)  # Resize image to 250x350 pixels
    image = ImageOps.invert(image)  # Invert the colors (black on white background)
    
    # Convert to numpy array
    pixel_data = np.array(image)
    
    # Flatten the image into a 1D array (for scikit-learn)
    pixel_data = pixel_data.flatten()

    # Apply PCA transformation to match the training data format
    pixel_data = pca.transform([pixel_data])  # PCA transformation
    
    return pixel_data


# Function to predict the digit
def predict_digit(image_path):
    # Process the image
    pixel_data = process_image(image_path)

    # Predict using the trained model
    prediction = model.predict(pixel_data)
    
    return prediction[0]

# GUI Application
class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")

        self.label = tk.Label(self.root, text="Upload a hand-drawn digit image (250x350)", font=("Helvetica", 14))
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self.root, text="Upload Image", font=("Helvetica", 12), command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="Prediction: ", font=("Helvetica", 12))
        self.result_label.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            try:
                prediction = predict_digit(file_path)
                self.result_label.config(text=f"Prediction: {prediction}")
            except Exception as e:
                messagebox.showerror("Error", "Failed to make a prediction. Please try again.")
                print(str(e))

# Main function to run the GUI
def main():
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
