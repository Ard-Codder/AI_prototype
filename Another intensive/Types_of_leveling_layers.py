'''Слой MaxPooling2D'''

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(16, (3, 3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())
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

'''Слой AveragePooling2D'''

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(16, (3, 3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(AveragePooling2D((4, 4)))
model.add(Flatten())
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



'''Слой GlobalMaxPooling2D'''

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(16, (3,3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(GlobalMaxPooling2D())
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

worker.train_model(
    model,
    epochs = 20,
    batch_size = 64,
)

worker.layers_demonstration()


'''Слой GlobalAveragePooling2D'''

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(16, (3,3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
model.add(AveragePooling2D((4, 4)))
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

worker.train_model(
    model,
    epochs = 20,
    batch_size = 64,
)

worker.layers_demonstration()