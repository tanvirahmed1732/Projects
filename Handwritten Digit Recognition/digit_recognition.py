import numpy as np
import cv2
from weka.core import jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
from weka.core.serialization import read
from weka.core.dataset import Instance, Instances
from PIL import Image

# Start the JVM
if not jvm.started:
    jvm.start()

# Load the model
model_path = r"D:\Code\GitHub\Projects\Digit Recognition\digit_recognition.model"
classifier = read(model_path)

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # grayscale
    img = cv2.resize(img, (350, 250))  # Note: width=350, height=250
    img = img.astype(np.float32) / 255.0  # normalize pixels
    flat_img = img.flatten()  # flatten to length 87,500
    return flat_img


# Function to predict the digit
def predict_digit(image_path):
    instance_data = preprocess_image(image_path)

    loader = Loader("weka.core.converters.ArffLoader")
    arff_path = r"D:\Code\GitHub\Projects\Digit Recognition\images.arff"
    dataset = loader.load_file(arff_path)
    dataset.class_is_last()

    inst = Instance.create_instance(instance_data.tolist())
    new_dataset = Instances.template_instances(dataset, 1)
    new_dataset.add_instance(inst)
    new_dataset.class_is_last()

    java_instance = new_dataset.get_instance(0).jobject
    prediction = classifier.classifyInstance(java_instance)
    return prediction


# Main function
if __name__ == "__main__":
    image_path = r"D:\Code\GitHub\Projects\Digit Recognition\Images\G4_E0.jpg"

    # Predict the digit
    predicted_digit = predict_digit(image_path)

    print(f"The predicted digit is: {predicted_digit}")

# Stop the JVM after prediction is done
if jvm.started:
    jvm.stop()
