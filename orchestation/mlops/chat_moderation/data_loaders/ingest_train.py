import os
import pandas as pd
import gdown
import tempfile

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)
    return destination

@data_loader
def load_train_csv(**kwargs) -> pd.DataFrame:
    file_id = '1ZrATssvboCgeJW6gIhGJPEvhnqtvTdWa'
    csv_file_path = tempfile.mktemp(suffix='.csv')
    download_file_from_google_drive(file_id, csv_file_path)
    try:
        df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
        print(f"Loaded DataFrame shape for train CSV: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading train CSV file: {e}")
        return pd.DataFrame()

@test
def test_train_csv(train_df) -> None:
    assert train_df is not None, 'The output is undefined'
    assert isinstance(train_df, pd.DataFrame), 'Output should be a DataFrame'
    assert not train_df.empty, 'The DataFrame should not be empty'
    print(f"Train CSV DataFrame is valid with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
