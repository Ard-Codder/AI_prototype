'''Слой UpSampling2D'''

# Создание модели
model = Sequential()

# Добавление слоев

model.add(Conv2D(8, (3, 3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(UpSampling2D((3, 3)))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(AveragePooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

worker.train_model(
    model,
    epochs=20,
    batch_size=64,
)

worker.layers_demonstration()

'''Слой Conv2DTranspose'''

# Создание модели
model = Sequential()

# Добавление слоев

model.add(Conv2D(8, (3, 3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(AveragePooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

worker.train_model(
    model,
    epochs=20,
    batch_size=64,
)

worker.layers_demonstration()

worker.train_model(
    model,
    epochs=1000,
    batch_size=64,
)

worker.layers_demonstration()
