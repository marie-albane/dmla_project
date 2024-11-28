import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from dmla.params import
from dmla.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from dmla.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from dmla.ml_logic.preprocessor import preprocess_features
from dmla.ml_logic.registry import load_model, save_model, save_results
from dmla.ml_logic.registry import mlflow_run, mlflow_transition_model

def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw data from BigQuery using `get_data_with_cache`
    # Retrieve data using `get_data_with_cache`
    # Process data
    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()


    print("✅ preprocess() done \n")

@mlflow_run
def train(
        min_date:str = '2009-01-01',
        max_date:str = '2015-01-01',
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)


    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!
    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
     # Create (X_train_processed, y_train, X_val_processed, y_val)
    # Train model using `model.py`
    # Save results on the hard drive using taxifare.ml_logic.registry
    # Save model weight on the hard drive (and optionally on GCS too!)
    # The latest model should be moved to staging

    print("✅ train() done \n")



@mlflow_run
def evaluate

    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    print("✅ evaluate() done \n")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    print("\n⭐️ Use case: predict")

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")



if __name__ == '__main__':
    preprocess(min_date='2009-01-01', max_date='2015-01-01')
    train(min_date='2009-01-01', max_date='2015-01-01')
    evaluate(min_date='2009-01-01', max_date='2015-01-01')
    pred()
