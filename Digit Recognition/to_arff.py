import os
import numpy as np
from scipy.io import arff
from PIL import Image

# Directory containing the images
image_dir = r'D:\Code\GitHub\Projects\Digit Recognition\Images'

# List of groups and digits
groups = [f'G{i}' for i in range(1, 13)]
digits_english = [f'E{digit}' for digit in range(10)]  # English digits E0 to E9
digits_bangla = [f'B{digit}' for digit in range(10)]  # Bangla digits B0 to B9

# Function to read and flatten images
def process_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((250, 350))  # Resize image to 250x350
    return np.array(image).flatten()  # Flatten image into 1D array

# List to hold all data
data = []

# Loop through each group and digit
for group in groups:
    for digit in range(10):  # For each digit
        for language in ['E', 'B']:  # 'E' for English and 'B' for Bangla
            image_name = f'{group}_{language}{digit}.jpg'  # Assuming JPG extension
            image_path = os.path.join(image_dir, image_name)
            
            # Process the image and get the flattened pixel values
            if os.path.exists(image_path):
                flattened_image = process_image(image_path)
                # Append the flattened image and the corresponding class label
                data.append(np.append(flattened_image, f'{language}{digit}'))

# Convert data to numpy array
data = np.array(data) 

# Define the ARFF header with both Bangla and English digits
attributes = [f'pixel{i+1}' for i in range(data.shape[1] - 1)]  # Create attribute names for each pixel
attributes.append('@attribute class {B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,E0,E1,E2,E3,E4,E5,E6,E7,E8,E9}')  # Class attribute (Bangla and English digits)

# Define the ARFF file structure
arff_data = []
arff_data.append('@relation images')  # Relation name
arff_data.extend(attributes)  # Add pixel attributes and class
arff_data.append('@data')  # Start data section

# Add the data (flattened image pixels and class labels)
for row in data:
    arff_data.append(','.join(map(str, row)))

# Write to an ARFF file
arff_file_path = 'images.arff'
with open(arff_file_path, 'w') as f:
    f.write('\n'.join(arff_data))

print(f"ARFF file saved at: {arff_file_path}")
