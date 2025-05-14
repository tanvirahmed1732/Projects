import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from PIL import Image
import numpy as np

# Start the JVM (Java Virtual Machine) for Weka
jvm.start()

# Load the Weka model
model = Classifier.forPath("digit_recognition.model")  # Path to your saved model

# Function to process the image: Convert to grayscale, resize, and flatten
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((250, 350))  # Resize to 250x350
    return np.array(image).flatten()  # Flatten the image into 1D array

# Function to predict the digit
def predict_digit(image_path):
    # Process the image
    flattened_image = process_image(image_path)

    # Convert the flattened image to Weka format (Instance)
    # Create a Weka instance with the same attributes as the ARFF file
    loader = Loader(classname="weka.core.converters.ArffLoader")
    dataset = loader.loadFile("images.arff")  # Load ARFF file used for training the model
    dataset.setClassIndex(dataset.numAttributes() - 1)  # Set the class attribute (digit label)

    # Create an instance for prediction (new data point)
    instance = dataset.get(0).copy()  # Copy the first instance
    for i in range(len(flattened_image)):
        instance.setValue(i, flattened_image[i])

    # Predict using the model
    prediction = model.classifyInstance(instance)
    class_label = dataset.classAttribute().value(int(prediction))  # Get the class label
    
    return class_label

# Test the function with an image
image_path = "path_to_your_handwritten_digit_image.jpg"  # Provide the path to the handwritten digit image
predicted_digit = predict_digit(image_path)
print(f"The predicted digit is: {predicted_digit}")

# Stop the JVM when done
jvm.stop()
