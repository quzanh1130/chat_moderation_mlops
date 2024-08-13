import os
import gdown
import zipfile
import tempfile

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)
    return destination

def check_file_content(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(4)
    if header != b'PK\x03\x04':  # Magic number for zip files
        raise ValueError("The file is not a valid zip file.")
    return True

def extract_zip(file_path):
    if check_file_content(file_path):
        with zipfile.ZipFile(file_path) as z:
            extract_path = tempfile.mkdtemp()  # Create a temporary directory
            z.extractall(extract_path)
            return extract_path, z.namelist()

@data_loader
def load_vector_zip(**kwargs) -> list:
    file_id = '1_3a43keyvHmOpZzH02Po9Gm8mHuFmw0Z'
    vec_zip_path = tempfile.mktemp(suffix='.zip')
    download_file_from_google_drive(file_id, vec_zip_path)
    try:
        extract_path, vec_files = extract_zip(vec_zip_path)
        print(f"Extracted vec_zip files: {vec_files}")
        return vec_files
    except ValueError as e:
        print(f"Error processing vec_zip file: {e}")
        return []

@test
def test_vector_files(vec_files) -> None:
    assert vec_files is not None, 'The output is undefined'
    print(type(vec_files))
    assert len(vec_files) > 0, 'The list of vector files should not be empty'
    print(f"Vector files: {vec_files}")
