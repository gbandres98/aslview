from aslview.data import prepare_data
from aslview.model import create_model
from aslview.training import train_model
from glob import glob

TRAIN_DIR = './data/asl_alphabet_train'
TEST_DIR = './data/asl_alphabet_test'
classes = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
classes_no = len(classes)

prepare_data(train_dir=TRAIN_DIR, test_dir=TEST_DIR)

model = create_model(classes_no=classes_no)

train_model(model)