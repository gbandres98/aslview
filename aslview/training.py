from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def train_model(model):

    # Esto no daba buenos resultados
    # datagen = ImageDataGenerator(  
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')

    batch_size = 64

    # Generador de imágenes de Keras
    generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.1,  # Tomamos el 10% de las imagenes para validacion
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    train_generator = generator.flow_from_directory(
            './data/asl_alphabet_train',
            target_size=(64, 64),
            batch_size=batch_size,
            shuffle=True,
            subset="training")
			
    validation_generator = generator.flow_from_directory(
            './data/asl_alphabet_train',
            target_size=(64, 64),
            batch_size=batch_size,
            subset="validation")

	# Configuracion necesaria para utilizar la RTX 2080
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)    
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

	# Entrenamos el modelo
    model.fit_generator(
        train_generator,
        epochs=6,
        validation_data=validation_generator,
        steps_per_epoch=1500,
        validation_steps=150
        )

    model.save_weights('pesos.h5') # Guardamos los pesos entrenados
    print('weights saved')
