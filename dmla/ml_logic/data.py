import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from dmla.params import *

#Fonction pour récupérer les y en fonction du dossier

def target_y(relative_path="data/raw_data/training"):

    """Fonction pour générer les y sous forme de DataFrame pour le dossier souhaité"""

    y = 0

    #DOSSIER RAW_DATA
    #Dossier training
    if relative_path == "data/raw_data/training":
        path = os.path.join("data/raw_data/RFMiD_Training_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        y = data["ARMD"]

    #Dossier validation
    elif relative_path == "data/raw_data/validation":
        path = os.path.join("data/raw_data/RFMiD_Validation_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        y = data["ARMD"]

    #Dossier testing
    elif relative_path == "data/raw_data/testing":
        path = os.path.join("data/raw_data/RFMiD_Testing_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        y = data["ARMD"]

    #DOSSIER 100_DATA
    #Dossier training
    elif relative_path == "data/100_data/training":
        path = os.path.join("data/raw_data/RFMiD_Training_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        nb_sample = int(len(os.listdir(relative_path))/2)
        data1 = data[data["ARMD"]==1].head(nb_sample)["ARMD"]
        data2 = data[data["ARMD"]==0].head(nb_sample)["ARMD"]
        y = pd.concat([data1, data2], ignore_index=True)


    #Dossier validation
    elif relative_path == "data/100_data/validation":
        path = os.path.join("data/raw_data/RFMiD_Validation_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        nb_sample = int(len(os.listdir(relative_path))/2)
        data1 = data[data["ARMD"]==1].head(nb_sample)["ARMD"]
        data2 = data[data["ARMD"]==0].head(nb_sample)["ARMD"]
        y = pd.concat([data1, data2], ignore_index=True)

    #Dossier testing
    elif relative_path == "data/100_data/testing":
        path = os.path.join("data/raw_data/RFMiD_Testing_Labels.csv")
        data = pd.read_csv(path)
        data = data.set_index("ID")
        y = data["ARMD"]
        nb_sample = int(len(os.listdir(relative_path))/2)
        data1 = data[data["ARMD"]==1].head(nb_sample)["ARMD"]
        data2 = data[data["ARMD"]==0].head(nb_sample)["ARMD"]
        y = pd.concat([data1, data2], ignore_index=True)

    else :
        print("Le dossier n'existe pas")


    return y






# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

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
