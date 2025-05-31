import tkinter as tk #for gui
from tkinter import filedialog, messagebox #For opening file picker and showing error popups
import numpy as np
from PIL import Image #to process images
import os
import subprocess #to run the Weka model from Python

def process_image(image_path):
    image = Image.open(image_path).convert('L')  # convert to grayscale
    image = image.resize((250, 350))

    threshold = 128
    image = image.point(lambda x: 0 if x < threshold else 255, 'L') #fixing the non 0,255 pixels to 0,255

    pixel_data = np.array(image).flatten() #1D Array
    return pixel_data



# 1D pixel values to arff
def create_arff_data(pixel_data, arff_path):
    #attributes
    attributes = [f'@attribute pixel{i+1} numeric' for i in range(250 * 350)]  # 250x350 pixels
    class_attr = '@attribute class {B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, E0, E1, E2, E3, E4, E5, E6, E7, E8, E9}'  # Class labels

    # ARFF header
    arff_data = ['@relation images']
    arff_data.extend(attributes)
    arff_data.append(class_attr)
    arff_data.append('@data')

    pixel_values = ','.join(map(str, pixel_data))
    arff_data.append(f'{pixel_values},?')  # Class is unknown (?) to be predicted

    with open(arff_path, 'w') as f:
        f.write('\n'.join(arff_data))


#to predict the digit from the image using Weka CLI
def predict_digit(image_path):
    arff_path = 'temp_image.arff'
    pixel_data = process_image(image_path)
    create_arff_data(pixel_data, arff_path)

    weka_jar = r'D:\Software\Weka-3-9-6\weka.jar' #weka jar file
    model_path = r"D:\Code\GitHub\Projects\Digit Recognition\digit_recognition.model" #weka trained model

    #Runs the Weka model using the command line
    command = [
        'java', '-cp', weka_jar,
        'weka.classifiers.trees.RandomForest',
        '-l', model_path,
        '-T', arff_path,
        '-p', '0'
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # print("=== RAW Weka Output ===")
    # print(result.stdout)
    # print("=== END OF OUTPUT ===")

    if result.returncode == 0:
        output = result.stdout.strip()
        prediction_started = False
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("inst#"):
                prediction_started = True
                continue
            if prediction_started and line:
                parts = line.split()
                if len(parts) >= 3 and ':' in parts[2]:
                    predicted_class = parts[2].split(":")[1] 
                    return predicted_class
        return "Prediction not found"
    else:
        print("Error:", result.stderr)
        return "Error in prediction"


# GUI
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
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                prediction = predict_digit(file_path)
                if prediction and not prediction.startswith("Error"):
                    self.result_label.config(text=f"Prediction: {prediction}")
                else:
                    messagebox.showerror("Prediction Failed", prediction)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred:\n{str(e)}")



# Main function to run the GUI
def main():
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
