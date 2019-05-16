from aslview.data import prepare_data
from aslview.model import create_model
from aslview.training import train_model
from keras.preprocessing import image
from glob import glob
import cv2
import numpy as np
import os

TRAIN_DIR = './data/asl_alphabet_train'
TEST_DIR = './data/asl_alphabet_test'
classes = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
classes_no = len(classes)

prepare_data(train_dir=TRAIN_DIR, test_dir=TEST_DIR)
print(classes)
model = create_model(classes_no=classes_no)

model.load_weights('hola.h5')

#img = cv2.imread('./data/asl_alphabet_test/N/N_test.jpg')
#img = cv2.resize(img,(64,64))
#img = np.reshape(img,[1,64,64,3])
#img = np.expand_dims(img,axis=0)


images = []
for img in os.listdir('./data/test/'):
    #img = image.load_img(img, target_size=(64, 64))
    img = cv2.imread('./data/test/'+img)
    img = cv2.resize(img,(64,64))
    #img = img.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)


res = model.predict_classes(images)

for i in range(len(res)):
    print(classes[res[i]])