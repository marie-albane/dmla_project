import pandas as pd
import os
import shutil


# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

from dmla.params import *


def copier_coller(nb_images_par_classe=100,set_data="training",): #Fait le 26.11.2024 - Michel


# Fonction pour copier/coller des images depuis le training set:
# - Préparation du nombre d'images à séléctionner, des chemins source et cible
# - Vérification de la création des chemins
# - Copie/colle

    #Choix du set de données
    path = os.path.join("data","raw_data","RFMiD_"+set_data.title()+"_Labels.csv")
    #path_training_set = (os.getcwd(),"data\raw_data\RFMiD_Training_Labels.csv") #regarder pour les slachs
    data = pd.read_csv(path)
    data = data.set_index("ID")

    #Dossier source training
    dossier_source = os.path.join(os.getcwd(), "data","raw_data",set_data.lower())

    #Dossier cible training
    dossier_cible = os.path.join(os.getcwd(), "data","100_data",set_data.lower())

    #Lister noms des images avec DLMLA
    positive_set = data[data["ARMD"]==1].head(nb_images_par_classe)
    positive_list = list(positive_set.index)

    #Lister noms des images avec DLMLA
    negative_set = data[data["ARMD"]==0].head(nb_images_par_classe)
    negative_list = list(negative_set.index)

    #Réunir les 2 listes
    images_list = positive_list + negative_list


    # Compteur du nb images
    compteur = 0
    introuvable = ""

    # Vérifier si le dossier cible existe, sinon le créer
    if not os.path.exists(dossier_cible):
        os.makedirs(dossier_cible)

    # Parcourir la liste des noms de fichiers
    for nom_fichier in images_list:
        # Construire le chemin complet des fichiers source et cible
        chemin_source = os.path.join(dossier_source, f"{nom_fichier}.png")
        chemin_cible = os.path.join(dossier_cible, f"{nom_fichier}.png")

        # Vérifier si le fichier existe dans le dossier source
        if os.path.exists(chemin_source):
            # Copier le fichier dans le dossier cible
            shutil.copy(chemin_source, chemin_cible)
            compteur += 1
            #print(f"Fichier copié : {chemin_source} -> {chemin_cible}")
        else:
            introuvable += {chemin_source}

    print(f"✅Total d'images copiées : {compteur} / {len(images_list)} pour {set_data}")
    if len(introuvable)!=0:
        print(f"Attention voici la liste des images non-copiées {introuvable}")

if __name__ == '__main__':
    copier_coller(50,"validation")



# """Code de taxifare pour inspiration"""

# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Clean raw data by
#     - assigning correct dtypes to each column
#     - removing buggy or irrelevant transactions
#     """
#     # Compress raw_data by setting types to DTYPES_RAW
#     df = df.astype(DTYPES_RAW)

#     # Remove buggy transactions
#     df = df.drop_duplicates()  # TODO: handle whether data is consumed in chunks directly in the data source
#     df = df.dropna(how='any', axis=0)

#     df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
#                     (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]

#     df = df[df.passenger_count > 0]
#     df = df[df.fare_amount > 0]

#     # Remove geographically irrelevant transactions (rows)
#     df = df[df.fare_amount < 400]
#     df = df[df.passenger_count < 8]

#     df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
#     df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
#     df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
#     df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

#     print("✅ data cleaned")

#     return df

# def get_data_with_cache(
#         gcp_project:str,
#         query:str,
#         cache_path:Path,
#         data_has_header=True
#     ) -> pd.DataFrame:
#     """
#     Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
#     Store at `cache_path` if retrieved from BigQuery for future use
#     """
#     if cache_path.is_file():
#         print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
#         df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
#     else:
#         print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
#         client = bigquery.Client(project=gcp_project)
#         query_job = client.query(query)
#         result = query_job.result()
#         df = result.to_dataframe()

#         # Store as CSV if the BQ query returned at least one valid line
#         if df.shape[0] > 1:
#             df.to_csv(cache_path, header=data_has_header, index=False)

#     print(f"✅ Data loaded, with shape {df.shape}")

#     return df

# def load_data_to_bq(
#         data: pd.DataFrame,
#         gcp_project:str,
#         bq_dataset:str,
#         table: str,
#         truncate: bool
#     ) -> None:
#     """
#     - Save the DataFrame to BigQuery
#     - Empty the table beforehand if `truncate` is True, append otherwise
#     """

#     assert isinstance(data, pd.DataFrame)
#     full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
#     print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

#     # Load data onto full_table_name



#     # 🎯 HINT for "*** TypeError: expected bytes, int found":
#     # After preprocessing the data, your original column names are gone (print it to check),
#     # so ensure that your column names are *strings* that start with either
#     # a *letter* or an *underscore*, as BQ does not accept anything else

#     new_columns =[]
#     for column in data.columns:
#         new_columns.append(f"_{str(column)}")

#     data.columns = new_columns
#     # YOUR CODE HERE
#     client = bigquery.Client()

#     if truncate:
#         write_mode = "WRITE_TRUNCATE"
#     else:
#         write_mode = "WRITE_APPEND"

#     job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

#     job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
#     result = job.result()

#     return result

#     print(f"✅ Data saved to bigquery, with shape {data.shape}")
