import streamlit as st
import requests
from PIL import Image

# Titre de la page
st.title('Hakuna Macula : le testeur de DMLA')

# Introduction
st.markdown("""
Un modèle performant porté par une équipe experte pour détecter ou non (de préférence) votre DMLA
""")

url = "https://XXXXXXXXXXXXXXXXXXXXXX.com/predict"  # Remplace avec l'URL de ton API

###### TELECHARGER UNE IMAGE ######
# Créer un formulaire pour le téléchargement de l'image
uploaded_file = st.file_uploader("Télécharger une image", type=["png"])

if uploaded_file is not None:
    # Charger l'image et l'afficher
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)


#####CHARGER UNE IMAGE EN LOCAL######
# Chemin vers l'image locale
image_path = "path/to/your/image.png"

# Charger l'image
image = Image.open(image_path)

# Afficher l'image
st.image(image, caption="Image affichée", use_column_width=True)




# Exemple du dico de taxifare
# params = {
#     'date_heure': date_heure,
#     'pickup_longitude': pickup_longitude,
#     'pickup_latitude': pickup_latitude,
#     'dropoff_longitude': dropoff_longitude,
#     'dropoff_latitude': dropoff_latitude,
#     'passenger_count': passenger_count
# }

# Faire une requête à l'API
response = requests.post(url, json=params)

# Récupérer la prédiction et l'afficher
if response.status_code == 200:
    prediction = response.json()
    st.write(f"Prédiction de tarif: {prediction['fare']} €")
else:
    st.error("Erreur lors de l'appel à l'API.")
