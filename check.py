from aslview.data import prepare_data
from aslview.model import create_model
from aslview.training import train_model
from glob import glob
import cv2
import numpy as np

TRAIN_DIR = './data/asl_alphabet_train'
TEST_DIR = './data/asl_alphabet_test'
classes = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
classes_no = len(classes)

prepare_data(train_dir=TRAIN_DIR, test_dir=TEST_DIR)

model = create_model(classes_no=classes_no)

model.load_weights('hola.h5')

img = cv2.imread('./data/asl_alphabet_test/B_test.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])
#img = np.expand_dims(img,axis=0)

classes = model.predict_classes(img)

print(classes)