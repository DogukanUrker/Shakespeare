OBJECT_NAME = "object"  # Name of the object to be detected

NOT_OBJECT_NAME = "notObject"  # Name of the not object to be detected

DATA_PATH = "data/"  # Path to the data folder

OBJECT_IMAGES_PATH = (
    DATA_PATH + OBJECT_NAME + "/"
)  # Path to the folder containing images of the object to be detected

NOT_OBJECT_IMAGES_PATH = (
    DATA_PATH + NOT_OBJECT_NAME + "/"
)  # Path to the folder containing images of the object not to be detected

TEST_IMAGES_PATH = DATA_PATH + "test/"  # Path to the folder containing test images

MODEL_NAME = "resnet"  # Name of the model to be used for detection (resnet, efficientnet, vgg, densenet, mobilenet)

PKL_FILE_PATH = "pkl/"  # Path to the folder where the pkl file will be saved

PKL_FILE_NAME = (
    PKL_FILE_PATH + OBJECT_NAME + "_" + MODEL_NAME + ".pkl"
)  # Name of the pkl file to be saved
