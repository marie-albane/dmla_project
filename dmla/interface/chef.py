import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from dmla.params import
from dmla.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from dmla.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model, modelisation
from dmla.ml_logic.preprocessor import preprocess_features, preprocess_images
from dmla.ml_logic.registry import load_model, save_model, save_results


    # print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    # print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

# préparation des données
X_train_proc, y_train_proc = preprocess_images(path='data/raw_data/training', data_set = "training")
X_val_proc, y_val_proc = preprocess_images(path='data/raw_data/validation',data_set = "validation")
X_test_proc, y_test_proc = preprocess_images(path='data/raw_data/testing',data_set = "testing")

model, params, score = modelisation(X_train_proc, y_train_proc, X_val_proc, y_val_proc, X_test_proc, y_test_proc)

# save(model, params, score)


if __name__ == '__main__':
    # preprocess(min_date='2009-01-01', max_date='2015-01-01')
    # train(min_date='2009-01-01', max_date='2015-01-01')
    # evaluate(min_date='2009-01-01', max_date='2015-01-01')
    # pred()
