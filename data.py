import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt
# import plotly.subplot as make_subplots 
import copy 
import os

'''
Introduction:

Motivation for the study:

This research seeks to explore the application of CNN architecure in detecting early stage cancer through data obtained from the Atomic Force Micrsopy (AFM). Early stage bladder cancer can be hard to detect even under expert's eyes and this research is not meant to replace surgeons and medical experts but meant to help speeden and reduce errors (false negatives and false positives) associated with early bladder cancer detection and give patients a trusted non-invasive way to encourage regular testing.

Problem Statement: The False Positive Rate and AUC of ROC curve is generally not very optimal for most early bladder cancer prediction and that is why using machine learning techniques is on the uprise.

Also, patients feel more motivated for non-invasive testing when it comes to regular testing and bladder cancer falls in this category, therefore, deep learning through a non-invasive bladder cancer detection method such as AFM could be very useful in boosting testing participation across potentially prone people.

Study aim: The aim of this study is to investigate the potential of using Pytorch Deep learning module:

For detecting to beat benchmark accuracy of experts.
'''


"""
Dataset analysis and exploration
"""

# Load the dataset
bladder_cancer_data = pd.read_csv('/Users/mac/Bladder_cancer/data/AASCII.zip', compression='zip') #on_bad_lines='skip'

# Print the head of the dataset
print(bladder_cancer_data.head())
# Import the dataset
bladder_cancer_data = pd.read_csv("/AASCII.csv")
print(bladder_cancer_data.head())


# Store the file path
os.listdir("/content/drive/MyDrive/Bladder_cancer/")

# Check for data informmation
print(bladder_cancer_data.describe())
print(bladder_cancer_data.shape)

# Check for duplicates entries of entries and other inconsistencies
# This will be a check for any duplicates and errors in the dataset, and if it is nencessary we drop them.

bladder_cancer_data_duplicates = bladder_cancer_data[bladder_cancer_data.duplicated(keep=False)]
print(bladder_cancer_data_duplicates)

# As of now, there are no duplicates in our data since the bladder_cancer_duplicates is zero.


# Data transformation:

# This part involves seperating the data into its targets and object values in order to enable us to compare
# the predicted outputs with the actual output

# So far, we can see that our actual outputs go on a horizontal row in the dataset, so let's seperate that
bladder_cancer_data_transposed = bladder_cancer_data.T
print(bladder_cancer_data_transposed)

bladder_cancer_data_transposed["label"] = bladder_cancer_data_transposed[1]
bladder_cancer_data_transposed = bladder_cancer_data_transposed.drop(1, axis = 1)
print(bladder_cancer_data_transposed)


# Target feature class balance

# We do this to explore the balance of the dataset that has the bladder cancer (malignant cells) and those that don't (non-malignant cells).

# These are useful insights from this part of the data:
# * There are 50 tests samples with no trace of bladder cancer in the data of 133 individual test, though made up of 20 individuals.
# * There are 62 tests samples showing mild level of cancer detection among 133 tests.
# * There are 21 test samples showing malignant cancer cells as provided.

# Our goal is to get our CNN architecture to perform as close to these actual y-outputs as posssible.
bladder_cancer_data_transposed["label"].value_counts()

