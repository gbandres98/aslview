import kaggle
import zipfile
from pathlib import Path

def prepare_data(train_dir, test_dir):
    print('Checking existence of training dataset...')
    train_set = Path(train_dir)
    test_set = Path(test_dir)
    if (train_set.exists() and train_set.stat().st_size > 500 and
            test_set.exists() and test_set.stat().st_size > 500):
        print('Dataset exists')
    else:
        print('Dataset not found, downloading from Kaggle')
        download_dataset()
        unzip(train_dir)
        unzip(test_dir)

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
            dataset='grassknoted/asl-alphabet',
            path='./data',
            quiet=False, unzip=True
    )

def unzip(train_dir):
    zip_ref = zipfile.ZipFile(train_dir + '.zip', 'r')
    zip_ref.extractall('./data/')
    zip_ref.close()