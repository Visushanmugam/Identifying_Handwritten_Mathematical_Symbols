
"""This file using libraries"""
from keras._tf_keras.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D
from keras._tf_keras.keras.layers import BatchNormalization, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(82, activation="softmax"))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

traindata_gen = ImageDataGenerator(rescale=None,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validationdata_gen = ImageDataGenerator(rescale=1./255)


traindata = traindata_gen.flow_from_directory("dataset/train",
                                              target_size=(128, 128),
                                              batch_size=32,
                                              class_mode="categorical")
train_label = traindata.class_indices

validdata = validationdata_gen.flow_from_directory("dataset/valid",
                                              target_size=(128, 128),
                                              batch_size=32,
                                              class_mode="categorical")

valid_label = validdata.class_indices

model.fit(traindata, epochs=5,
                    validation_data=validdata)

model_json = model.to_json()
print(model_json)
with open("model.json", 'w') as modeljson:
    modeljson.write(model_json)
    model.save_weights('model.weights.h5')
    modeljson.close()
