import os  # Import the os module for interacting with the operating system
from PIL import Image  # Import the Image module from PIL for image processing
import torch  # Import the torch module for working with PyTorch
import torchvision.models as models  # Import the models module from torchvision for pre-trained models
import torchvision.transforms as transforms  # Import the transforms module from torchvision for data transformations
import joblib  # Import the joblib module for model serialization
import io  # Import the io module for handling byte streams
import ssl  # SSL certificate handling
from defaults import (
    MODEL_NAME,
    OBJECT_NAME,
    PKL_FILE_NAME,
    TEST_IMAGES_PATH,
)  # Import the MODEL_NAME, OBJECT_NAME, PKL_FILE_NAME, and TEST_IMAGES_PATH variables from the defaults module.

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Define the target image size for resizing
imageSize = (256, 256)

# Define the data preprocessing transformations (without augmentation for testing)
dataTransforms = transforms.Compose(
    [
        transforms.Resize(imageSize),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.256, 0.225]
        ),  # Normalize the image with specified mean and standard deviation
    ]
)

print(
    f"Testing the {MODEL_NAME} model..."
)  # Inform the user which model is being tested.

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

model.eval()  # Set the model to evaluation mode

# Remove the final classification layer from the model to use it as a feature extractor
featureExtractor = torch.nn.Sequential(*list(model.children())[:-1])


# Function to extract features from a single image using the pre-trained model
def extractFeatures(image):
    """
    Extracts features from an image using a feature extractor model.

    Args:
        image (torch.Tensor): The input image tensor.

    Returns:
        numpy.ndarray: The extracted features as a flattened numpy array.
    """
    # Apply transformations and add a batch dimension
    image = dataTransforms(image).unsqueeze(0)
    with torch.no_grad():  # Disable gradient computation
        # Extract features and flatten them
        feature = featureExtractor(image).numpy().flatten()
    return feature  # Return the extracted features


# Load the trained model from the .pkl file
pipeline = joblib.load(PKL_FILE_NAME)


# Function to predict if an image is an object or not using the extracted features
def prediction(image):
    """
    Predicts the class of an image using a trained model.

    Args:
        image (str): The path to the image file.

    Returns:
        bool: True if the predicted class is object (0), False otherwise.
    """
    img = Image.open(image).convert("RGB")  # Open the image and convert it to RGB
    feature = extractFeatures(img)  # Extract features from the image
    result = pipeline.predict([feature])  # Predict the class using the trained model
    return (
        True if result[0] == 0 else False
    )  # Return True if the class is object (0), otherwise False


# Function to predict if an image is an object or not given its file path
def predictImage(imagePath):
    """
    Predicts the image using the given image path.

    Args:
        imagePath (str): The path of the image to be predicted.

    Returns:
        str: The prediction result.

    """
    result = prediction(imagePath)  # Predict the image
    return result  # Return the prediction result


# Function to predict if an image is an object or not given its byte content
def predictImageFromBytes(imageBytes):
    """
    Predicts the image based on the provided byte content.

    Args:
        imageBytes (bytes): The byte content of the image.

    Returns:
        str: The prediction result.

    """
    img = io.BytesIO(imageBytes)  # Convert the byte content to a byte stream
    result = prediction(img)  # Predict the image
    return result  # Return the prediction result


# Function to predict if images in a directory are an object or not
def predictImagesInDirectory(directoryPath):
    """
    Predicts images in a given directory and returns the results.

    Args:
        directoryPath (str): The path to the directory containing the images.

    Returns:
        list: A list of tuples, where each tuple contains the filename and the prediction result for the corresponding image.
    """
    results = []  # Initialize an empty list to store the results
    for filename in os.listdir(
        directoryPath
    ):  # Iterate over the files in the directory
        filepath = os.path.join(
            directoryPath, filename
        )  # Get the full path of the file
        if os.path.isfile(filepath):  # Check if it is a file
            result = predictImage(filepath)  # Predict the image
            results.append((filename, result))  # Append the result to the list
    return results  # Return the list of results


predictions = predictImagesInDirectory(
    TEST_IMAGES_PATH
)  # Predict the images in the test directory

for filename, result in predictions:  # Iterate over the prediction results
    print(
        f"Image: {filename}, {OBJECT_NAME}: {result}"
    )  # Print the filename and prediction result
