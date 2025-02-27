from defaults import *  # Import everything from the defaults module, which likely includes default settings and paths.
import os  # Import the os module for interacting with the operating system.
import subprocess  # Import the subprocess module to run external commands.


def setupProject():
    """
    Sets up the project by creating necessary folders and installing required packages.
    """
    print(
        "Setting up the project..."
    )  # Inform the user that the project setup is starting.

    if not os.path.exists(
        DATA_PATH
    ):  # Check if the data folder path defined in defaults does not exist.
        os.mkdir(DATA_PATH)  # Create the data folder.
        print(
            "Data folder created!"
        )  # Inform the user that the data folder has been created.
    else:
        print(
            "Data folder already exists!"
        )  # Inform the user that the data folder already exists.

    if not os.path.exists(
        OBJECT_IMAGES_PATH
    ):  # Check if the {OBJECT_NAME} images folder path does not exist.
        os.mkdir(OBJECT_IMAGES_PATH)  # Create the {OBJECT_NAME} images folder.
        print(
            f"{OBJECT_NAME} images folder created!"
        )  # Inform the user that the {OBJECT_NAME} images folder has been created.
    else:
        print(
            f"{OBJECT_NAME} images folder already exists!"
        )  # Inform the user that the {OBJECT_NAME} images folder already exists.

    if not os.path.exists(
        NOT_OBJECT_IMAGES_PATH
    ):  # Check if the {NOT_OBJECT_NAME} images folder path does not exist.
        os.mkdir(NOT_OBJECT_IMAGES_PATH)  # Create the {NOT_OBJECT_NAME} images folder.
        print(
            f"{NOT_OBJECT_NAME} images folder created!"
        )  # Inform the user that the {NOT_OBJECT_NAME} images folder has been created.
    else:
        print(
            f"{NOT_OBJECT_NAME} images folder already exists!"
        )  # Inform the user that the {NOT_OBJECT_NAME} images folder already exists.

    if not os.path.exists(
        TEST_IMAGES_PATH
    ):  # Check if the test images folder path does not exist.
        os.mkdir(TEST_IMAGES_PATH)  # Create the test images folder.
        print(
            "Test images folder created!"
        )  # Inform the user that the test images folder has been created.
    else:
        print(
            "Test images folder already exists!"
        )  # Inform the user that the test images folder already exists.

    if not os.path.exists(
        PKL_FILE_PATH
    ):  # Check if the pkl folder path does not exist.
        os.mkdir(PKL_FILE_PATH)  # Create the pkl folder.
        print(
            "PKL folder created!"
        )  # Inform the user that the pkl folder has been created.
    else:
        print(
            "PKL folder already exists!"
        )  # Inform the user that the pkl folder already exists.

    subprocess.call(
        ["pip3", "install", "-r", "requirements.txt"]
    )  # Run the command to install all required packages from requirements.txt.

    if (
        __name__ == "__main__"
    ):  # Check if the script is being run directly (not imported as a module).
        print("Setup complete!")  # Inform the user that the setup is complete.


setupProject()  # Call the setupProject function to set up the project. (This line is necessary to run the setup when the script is executed.)
