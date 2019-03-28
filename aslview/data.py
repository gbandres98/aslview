import kaggle
import zipfile
from pathlib import Path

def prepare_data(train_dir: str):
    print('Checking existence of training dataset...')
    dataset = Path(train_dir)
    if (dataset.exists() and dataset.stat().st_size > 500):
        print('Dataset exists')
    else:
        print('Dataset not found, downloading from Kaggle')
        download_dataset()
        unzip(train_dir)

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
            dataset='grassknoted/asl-alphabet',
            path='./data',
            quiet=False, unzip=True
    )

def unzip(train_dir: str):
    zip_ref = zipfile.ZipFile(train_dir + '.zip', 'r')
    zip_ref.extractall('./data/')
    zip_ref.close()