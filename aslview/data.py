import kaggle
import zipfile
from pathlib import Path

def prepare_data():
    print('Checking existence of training dataset...')
    dataset = Path('./data/asl_alphabet_train')
    if (dataset.exists() and dataset.stat().st_size > 500):
        print('Dataset exists')
    else:
        print('Dataset not found, downloading from Kaggle')
        download_dataset()
        unzip()

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
            dataset='grassknoted/asl-alphabet',
            path='./data',
            quiet=False, unzip=True
    )

def unzip():
    zip_ref = zipfile.ZipFile('./data/asl_alphabet_train.zip', 'r')
    zip_ref.extractall('./data/')
    zip_ref.close()