from defaults import (
    MODEL_NAME,
)  # Import the MODEL_NAME variable from the defaults module.

print(f"Running the {MODEL_NAME} model...")  # Inform the user which model is being run.


def runModel():  # Define the runModel function.
    """
    Runs the specified model based on the value of MODEL_NAME.

    Raises:
        ValueError: If the specified model is not found.
    """
    if MODEL_NAME == "resnet":  # Check if the model name is "resnet".
        import models.resnet  # Import the resnet model from the models package.
    elif MODEL_NAME == "efficientnet":  # Check if the model name is "efficientnet".
        import models.efficientnet  # Import the efficientnet model from the models package.
    elif MODEL_NAME == "vgg":  # Check if the model name is "vgg".
        import models.vgg  # Import the vgg model from the models package.
    elif MODEL_NAME == "densenet":  # Check if the model name is "densenet".
        import models.densenet  # Import the densenet model from the models package.
    elif MODEL_NAME == "mobilenet":  # Check if the model name is "mobilenet".
        import models.mobilenet  # Import the mobilenet model from the models package.
    else:  # If the model name is neither "resnet" nor "efficientnet".
        raise ValueError(
            "Model not found!"
        )  # Raise an error indicating the model was not found.


runModel()  # Call the runModel function to run the appropriate model based on MODEL_NAME.
