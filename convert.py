import pandas as pd
import os
import numpy as np
from PIL import Image

# Recreate images
# We will convert the 512*512 data back into images and then we can use it for our CNN architecture


# Function to convert a single column of pixel values to an image
def convert_column_to_image(column, image_size=(512, 512)):

    # Convert column to numpy array and reshape into image_size, skipping the first item
    image_array = column[3:].values.reshape(image_size)
    # Normalize the pixel values to the range 0-255
    image_array = 255 * (image_array - image_array.min()) / (image_array.max() - image_array.min())

    # Create an image from the numpy array
    image = Image.fromarray(image_array.astype('uint8'), 'L')

    return image

# Path to the CSV file containing the DataFrame
csv_path = '/Users/mac/Bladder_cancer/data/AASCII.zip'
folder_path = '/Users/mac/Bladder_cancer/images'

# Create subdirectories for each type of cancer classification
classification_folders = {
    0: 'no_cancer',
    1: 'benign_cancer',
    2: 'malignant_cancer'
}

# Ensure the directories exist
for folder in classification_folders.values():
    os.makedirs(os.path.join(folder_path, folder), exist_ok=True)

# Function to create images from a DataFrame
def create_images_from_dataframe(csv_path, image_size=(512, 512)):
    # Load the DataFrame from a CSV
    bladder_cancer_data = pd.read_csv(csv_path, compression='zip')

    # Get the classification for each column from the designated row
    classification_row = bladder_cancer_data.iloc[1]


    # For each column in the DataFrame, create an image
    for i, (column_name, classification) in enumerate(zip(bladder_cancer_data.columns, classification_row)):

        # Convert column to image
        image = convert_column_to_image(bladder_cancer_data[column_name], image_size)

        # Define the subfolder based on the classification
        subfolder = classification_folders[classification]

        # Sve the imaage in the corresponding folder
        image_path = os.path.join(folder_path, subfolder, f'image_{i+1}.png')

        # Save the image to a file
        image.save(image_path)

# Create and save images
create_images_from_dataframe(csv_path)