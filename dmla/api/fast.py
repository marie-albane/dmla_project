import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dmla.ml_logic.registry import load_model
from dmla.ml_logic.preprocessor import load_and_process_random_image

app = FastAPI()

app.state.model, model_number = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict():      # 1
    """
    Prediction d'une image choisie au hasard dans le répertoire Testing
    """

    image_rgb, cropped_image, resized_image, normalized_image, image_name = load_and_process_random_image(wanted_dataset = "testing")
    image_with_batch = np.expand_dims(normalized_image, axis=0)

    #Charger le model avec la fonction best_model = load_model() et l image
    best_model = app.state.model

    result = best_model.predict(image_with_batch)

    if result < 0.5:
        dmla = 0
    else :
        dmla = 1

    return { 'DMLA (1=oui)' : dmla,
            'Prediction DMLA en %': round(result[0][0]*100,2),
            'numero image': image_name,
            'Numero model':model_number}

@app.get("/")
def root():
    return { 'Welcome': 'Bienvenue à la racine de l API DMLA' }
