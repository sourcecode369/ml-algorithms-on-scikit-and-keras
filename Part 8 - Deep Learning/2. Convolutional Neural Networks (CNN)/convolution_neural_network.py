#Convolution Neural Network

#Part - 1 : Building the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

 #Initializing the CNN
classifier = Sequential()

#Step - 1 Convolution Operation
classifier.add(Convolution2D(32, (3, 3), input_shape = (64,64,3),activation = 'relu'))
#Step - 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''#Adding a second convolution layer for better performance
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))'''

#Step - 3 Flattening
classifier.add(Flatten()) 

#Step - 4 Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part - 2 : Fitting the CNN to the images (Image Preprocessing to prevent Overfitting)
'''Keras Documentation -> Preprocessing -> Image Preprocessing -> flow_fromdirectory'''

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                         target_size=(64, 64),
                                         batch_size=32,
                                         class_mode='binary')
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=2,
                         validation_data=test_set,
                         validation_steps=2000)
'''
After second epoch:
    Epoch 2/25
8000/8000 [==============================] - 3079s 385ms/step - 
loss: 0.2054 - 
acc: 0.9163 - Training Set accuracy
val_loss: 0.8122 - 
val_acc: 0.7795 - Test Set Accuracy
'''
#Part - 3 - Making a new prediction
prediction = ' '
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset\single_prediction\cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction = 'Dog'
else:
    prediction = 'Cat'
