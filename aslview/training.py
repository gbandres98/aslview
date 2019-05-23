from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def train_model(model):
    # datagen = ImageDataGenerator(
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')

    batch_size = 64

    # this is the augmentation configuration we will use for training
    generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            validation_split=0.1
            )
            #rescale=1./255,
            #shear_range=0.2,
            #zoom_range=0.2,
            #horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    #test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = generator.flow_from_directory(
            './data/asl_alphabet_train',  # this is the target directory
            target_size=(64, 64),  # all images will be resized to 150x150
            batch_size=batch_size,  # since we use binary_crossentropy loss, we need binary labels
            shuffle=True,
            subset="training")
    # this is a similar generator, for validation data
    validation_generator = generator.flow_from_directory(
            './data/asl_alphabet_train',
            target_size=(64, 64),
            batch_size=batch_size,
            subset="validation")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
    
    
    sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())
    
    set_session(sess)  # set this TensorFlow session as the default session for Keras


    model.fit_generator(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        steps_per_epoch=1200,
        validation_steps=150
        )

    model.save_weights('hola.h5')
    print('weights saved')
    
    #weights('first_try.h5')  # always save your weights after training or during training
