import os
import numpy as np

##################  VARIABLES  ##################
TARGETED_IMAGES_X = int(os.environ.get('TARGETED_IMAGES_X',256))
TARGETED_IMAGES_Y = int(os.environ.get('TARGETED_IMAGES_Y',256))#MR_26/11
DATA_SIZE = os.environ.get("DATA_SIZE")
DATA_PATH= os.environ.get("DATA_PATH")

MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")


##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")



################## VALIDATIONS #################
