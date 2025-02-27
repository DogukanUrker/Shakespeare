# Import necessary libraries
import os  # For operating system related functionalities
import numpy as np  # Numerical operations library
from PIL import Image  # Python Imaging Library for image processing
import torch  # PyTorch deep learning framework
import torchvision.models as models  # Pre-trained models from torchvision
import torchvision.transforms as transforms  # Data transformations for images
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.preprocessing import (
    StandardScaler,
)  # Standardize features by removing the mean and scaling to unit variance
from sklearn.svm import SVC  # Support Vector Classifier from scikit-learn
from sklearn.pipeline import (
    make_pipeline,
)  # Constructing a pipeline from transformers and estimator
from sklearn.metrics import (
    accuracy_score,
)  # Evaluation metric for classification accuracy
import joblib  # Save and load Python objects (including sklearn models) to and from disk
import time  # Time-related functionalities
import ssl  # SSL certificate handling
from efficientnet_pytorch import EfficientNet  # EfficientNet model
from defaults import *  # Import constants from defaults.py

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

print("\n")  # Print newline for better readability

appStartTime = time.time()  # Record start time for app

# Define the directory where the dataset is stored and the class names
dataDir = DATA_PATH  # Directory containing the dataset
classes = [OBJECT_NAME, NOT_OBJECT_NAME]  # Class names for classification

# Define the target image size for resizing
imageSize = (256, 256)

# Define the data augmentation and preprocessing transformations
dataTransforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            256
        ),  # Crop the image to random size and aspect ratio
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(
            10
        ),  # Rotate the image by a random angle within [-10, 10] degrees
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.001
        ),  # Randomly change the brightness, contrast, saturation, and hue of the image
        transforms.RandomGrayscale(
            p=0.1
        ),  # Randomly convert the image to grayscale with a probability of 10%
        transforms.GaussianBlur(
            kernel_size=5, sigma=(0.1, 2.0)
        ),  # Apply Gaussian blur with a random kernel size and sigma
        transforms.ToTensor(),  # Convert the image to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image with specific mean and std
    ]
)


# Function to load images and their corresponding labels
def loadImages(dataDir, classes, imageSize):
    """
    Load images from the specified directory and resize them to the given image size.

    Args:
        dataDir (str): The directory path where the images are located.
        classes (list): A list of class names.
        imageSize (tuple): The desired size of the images after resizing.

    Returns:
        list: A list of tuples containing the loaded images and their corresponding labels.
    """
    images = []  # List to store images and their labels
    startTime = time.time()  # Record start time for loading images
    print(f"Loading images from '{dataDir}'...")  # Print progress message
    for label, cls in enumerate(classes):  # Iterate through each class
        clsDir = os.path.join(dataDir, cls)  # Path to class directory
        classImages = os.listdir(clsDir)  # List of images in the class directory
        print(
            f"Found {len(classImages)} images in class '{cls}'"
        )  # Print number of images found
        for filename in classImages:  # Iterate through each image in the class
            filepath = os.path.join(clsDir, filename)  # Full path to the image file
            if os.path.isfile(filepath):  # Check if it's a file (not a directory)
                img = Image.open(filepath).convert(
                    "RGB"
                )  # Open image and convert to RGB mode
                img = img.resize(imageSize)  # Resize image to predefined size
                images.append((img, label))  # Append tuple of image and its label
    endTime = time.time()  # Record end time for loading images
    print(
        f"Loaded {len(images)} images in {(endTime - startTime):.2f} seconds."
    )  # Print loading time
    return images  # Return list of loaded images and labels


# Load the images and their labels
imageLabelPairs = loadImages(dataDir, classes, imageSize)  # Call loadImages function

print("\n")  # Print newline for better readability

# Load a pre-trained MobileNet model for feature extraction
startTime = time.time()  # Record start time for loading MobileNet model
print(
    f"Loading pre-trained {MODEL_NAME} model..."
)  # Inform the user which model is being loading.
if MODEL_NAME == "resnet":  # Check if the model name is "resnet".
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif MODEL_NAME == "efficientnet":  # Check if the model name is "efficientnet".
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )  # Load pre-trained EfficientNet model
elif MODEL_NAME == "vgg":  # Check if the model name is "vgg".
    model = models.vgg16(
        weights=models.VGG16_Weights.DEFAULT
    )  # Load pre-trained VGG model
elif MODEL_NAME == "densenet":  # Check if the model name is "densenet".
    model = models.densenet201(
        weights=models.DenseNet201_Weights.DEFAULT
    )  # Load pre-trained DenseNet model
elif MODEL_NAME == "mobilenet":  # Check if the model name is "mobilenet".
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )  # Load pre-trained MobileNet model
else:  # If the model name is neither "resnet" nor "efficientnet".
    raise ValueError(
        "Model not found!"
    )  # Raise an error indicating the model was not found.
model.eval()  # Set model to evaluation mode (not training mode)
featureExtractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)  # Extract feature extractor from MobileNet model
endTime = time.time()  # Record end time for loading MobileNet model
print(
    f"Loaded {MODEL_NAME} model in {(endTime - startTime):.2f} seconds."
)  # Print loading time

print("\n")  # Print newline for better readability


# Function to extract features from images using the pre-trained model
def extractFeatures(images):
    """
    Extracts features from a list of images.

    Args:
        images (list): A list of image-label pairs.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the extracted features,
        and the second array contains the corresponding labels.

    """
    features = []  # List to store extracted features
    labels = []  # List to store corresponding labels
    print(f"Extracting features from {len(images)} images...")  # Print progress message
    for idx, (image, label) in enumerate(
        images, start=1
    ):  # Iterate through each image-label pair
        image = dataTransforms(image).unsqueeze(
            0
        )  # Apply data transformations and add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            feature = (
                featureExtractor(image).numpy().flatten()
            )  # Extract features and convert to numpy array
        features.append(feature)  # Append extracted feature to list
        labels.append(label)  # Append corresponding label to list
        if idx % 100 == 0 or idx == len(
            images
        ):  # Print progress every 100 images processed or at the end
            print(f"Processed {idx}/{len(images)} images...")  # Print progress message
    return np.array(features), np.array(
        labels
    )  # Convert lists to numpy arrays and return


# Extract features and labels from the images
startTime = time.time()  # Record start time for feature extraction
features, labels = extractFeatures(imageLabelPairs)  # Call extractFeatures function
endTime = time.time()  # Record end time for feature extraction
print(
    f"Extracted features and labels in {(endTime - startTime):.2f} seconds."
)  # Print extraction time

print("\n")  # Print newline for better readability

# Split the dataset into training and validation sets
startTime = time.time()  # Record start time for dataset splitting
print("Splitting dataset into train and validation sets...")  # Print progress message
XTrain, XVal, yTrain, yVal = train_test_split(
    features, labels, test_size=0.1, random_state=42
)  # Split dataset
endTime = time.time()  # Record end time for dataset splitting
print(
    f"Split dataset into train and validation sets in {(endTime - startTime):.2f} seconds."
)  # Print splitting time

print("\n")  # Print newline for better readability

# Create a pipeline with StandardScaler and SVM classifier
startTime = time.time()  # Record start time for pipeline creation
print(
    "Creating pipeline with StandardScaler and SVM classifier..."
)  # Print progress message
pipeline = make_pipeline(
    StandardScaler(), SVC(kernel="linear", probability=True)
)  # Create pipeline
endTime = time.time()  # Record end time for pipeline creation
print(
    f"Created pipeline in {(endTime - startTime):.2f} seconds."
)  # Print pipeline creation time

print("\n")  # Print newline for better readability

# Train the classifier on the training set
startTime = time.time()  # Record start time for training
print("Training SVM classifier...")  # Print progress message
pipeline.fit(XTrain, yTrain)  # Train pipeline on training set
endTime = time.time()  # Record end time for training
print(
    f"Trained classifier in {(endTime - startTime):.2f} seconds."
)  # Print training time

print("\n")  # Print newline for better readability

# Predict labels for the validation set
startTime = time.time()  # Record start time for prediction
print("Predicting labels for validation set...")  # Print progress message
yPred = pipeline.predict(XVal)  # Predict labels for validation set
endTime = time.time()  # Record end time for prediction
print(
    f"Predicted labels for validation set in {(endTime - startTime):.2f} seconds."
)  # Print prediction time

print("\n")  # Print newline for better readability

# Calculate the accuracy of the classifier
startTime = time.time()  # Record start time for accuracy calculation
accuracy = accuracy_score(yVal, yPred)  # Calculate accuracy
endTime = time.time()  # Record end time for accuracy calculation
print(
    f"Calculated accuracy in {(endTime - startTime):.2f} seconds."
)  # Print accuracy calculation time
print(f"Validation Accuracy: {accuracy:.4f}")  # Print validation accuracy

print("\n")  # Print newline for better readability

# Save the trained model to a file
startTime = time.time()  # Record start time for model saving
print("Saving trained model to file...")  # Print progress message
joblib.dump(pipeline, PKL_FILE_NAME)  # Save pipeline to file
endTime = time.time()  # Record end time for model saving
print(
    f"Saved trained model to file in {(endTime - startTime):.2f} seconds."
)  # Print model saving time

print("\n")  # Print newline for better readability

# Load the model from the file (for demonstration purposes)
startTime = time.time()  # Record start time for model loading
print("Loading model from file...")  # Print progress message
pipeline = joblib.load(PKL_FILE_NAME)  # Load pipeline from file
endTime = time.time()  # Record end time for model loading
print(
    f"Loaded model from file in {(endTime - startTime):.2f} seconds."
)  # Print model loading time

print("\n")  # Print newline for better readability

# Predict and print the first 10 predictions on the validation set
startTime = time.time()  # Record start time for prediction
print("Predicting first 10 samples from validation set…")  # Print progress message
print(pipeline.predict(XVal[:10]))  # Predict and print first 10 samples
endTime = time.time()  # Record end time for prediction
print(
    f"Predicted first 10 samples in {(endTime - startTime):.2f} seconds."
)  # Print prediction time

print("\n")  # Print newline for better readability

appEndTime = time.time()  # Record end time for app
print(
    f"Training and testing completed successfully in {(appEndTime - appStartTime):.2f} seconds."
)  # Print completion message

print("\n")  # Print newline for better readability
