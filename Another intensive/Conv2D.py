# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(8, (3,3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(4, (3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(worker.x_train,
          worker.y_train,
          validation_data=(worker.x_test, worker.y_test),
          batch_size=24,
          epochs=20,
          verbose=1)

worker.plot_graphic(history)

worker.layers_demonstration(model)




'''padding='valid'''

# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(8, (3,3), input_shape=(108, 192, 3), padding='valid', activation='relu'))
model.add(Conv2D(4, (3,3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(worker.x_train,
          worker.y_train,
          validation_data=(worker.x_test, worker.y_test),
          batch_size=24,
          epochs=20,
          verbose=1)

worker.plot_graphic(history)

worker.layers_demonstration(model)



