# @title сервисные функции
import gdown
import pandas as pd
from keras import backend as K
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten
from tensorflow.python.keras.optimizer_v2.nadam import Nadam  # был optimizer import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import seaborn as sns
import os
import random
import cv2
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Worker:
    url_ds = 'https://storage.yandexcloud.net/aiueducation/Content/base/l5/middle_fmr.zip'
    classes = {
        0: 'Феррари',
        1: 'Мерседес',
        2: 'Рено',
    }

    def __init__(self):
        pass

    def load_data(self):
        print(f'{bcolors.BOLD}Загрузка датасета:', end='')
        output = 'data.zip'
        gdown.download(self.__class__.url_ds, output, quiet=True)

        # Extract the downloaded file into the 'data' directory
        import zipfile
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.remove(output)
        print(f'{bcolors.OKGREEN} Done {bcolors.ENDC}')
        data_dir = 'data'
        num_images = 0
        num_images_by_category = {}
        for dir_name in sorted(os.listdir(data_dir)):
            if not os.path.isdir(os.path.join(data_dir, dir_name)):
                continue
            num_images_by_category[dir_name] = 0
            for fname in os.listdir(os.path.join(data_dir, dir_name)):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    num_images += 1
                    num_images_by_category[dir_name] += 1

        print(f'{bcolors.BOLD}Информация о датасете:{bcolors.ENDC}')
        print(f'  {bcolors.OKBLUE}размер:{bcolors.ENDC} {num_images} изображений')
        for i, (dir_name, num_images) in enumerate(num_images_by_category.items()):
            print(f'  {bcolors.OKBLUE}класс {self.__class__.classes[i]}:{bcolors.ENDC} {num_images} изображений')

    def show_samples(self):
        data_dir = 'data'
        for j, dir_name in enumerate(sorted(os.listdir(data_dir))):
            if not os.path.isdir(os.path.join(data_dir, dir_name)):
                continue

            image_files = [os.path.join(data_dir, dir_name, fname)
                           for fname in os.listdir(os.path.join(data_dir, dir_name))
                           if fname.endswith('.jpg') or fname.endswith('.png')]

            selected_files = random.sample(image_files, min(len(image_files), 5))

            fig, axs = plt.subplots(1, len(selected_files), figsize=(20, 3))
            for i, fname in enumerate(selected_files):
                img = plt.imread(fname)
                axs[i].imshow(img)
                axs[i].axis('off')
                axs[i].set_title(os.path.basename(fname))
            plt.suptitle(self.__class__.classes[j])
            plt.show()

    def create_sets(self, test_size=0.1, random_state=42):
        print(f'{bcolors.BOLD}Создание обучающей и проверочной выборки:', end='')
        data_dir = 'data'
        images = []
        labels = []
        for i, dir_name in enumerate(sorted(os.listdir(data_dir))):
            if not os.path.isdir(os.path.join(data_dir, dir_name)):
                continue
            for fname in os.listdir(os.path.join(data_dir, dir_name)):
                if fname.endswith('.jpg') or fname.endswith('.png'):
                    img = cv2.imread(os.path.join(data_dir, dir_name, fname))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (192, 108))
                    images.append(img)
                    labels.append(i)

        # Convert images and labels to numpy arrays
        X = np.array(images)
        y = to_categorical(np.array(labels), 3)

        # Split the dataset into training and validation sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=test_size,
                                                                                random_state=random_state)

        print(f'{bcolors.OKGREEN} Done {bcolors.ENDC}')
        print()
        print(f'{bcolors.BOLD}Размеры сформированных выборок:{bcolors.ENDC}')
        print(f'  обучающая выборка параметров', self.x_train.shape)
        print(f'  обучающая выборка меток классов', self.y_train.shape)
        print()
        print(f'  тестовая выборка параметров', self.x_test.shape)
        print(f'  тестовая выборка меток классов', self.y_test.shape)

    def plot_graphic(self, history):
        # Обучение модели
        f, ax = plt.subplots(1, 2, figsize=(30, 8))
        ax[0].plot(history.history['accuracy'],
                   label='Доля верных ответов на обучающем наборе')
        ax[0].plot(history.history['val_accuracy'],
                   label='Доля верных ответов на проверочном наборе')
        ax[0].set_xlabel('Эпоха обучения')
        ax[0].set_ylabel('Доля верных ответов')

        ax[1].plot(history.history['loss'],
                   label='Ошибка на обучающем наборе')
        ax[1].plot(history.history['val_loss'],
                   label='Ошибка на проверочном наборе')
        ax[1].set_xlabel('Эпоха обучения')
        ax[1].set_ylabel('Ошибка')
        plt.show()

    def layers_demonstration(self, model):
        self.model = model
        print(f'{bcolors.BOLD}{bcolors.OKBLUE}Демонстрация работы слоев нейронной сети{bcolors.ENDC}')
        print(f'Для примера возьмем случайное изображение из тестовой выборки:')
        data_dir = 'data'
        idx = np.random.choice(self.x_test.shape[0])
        plt.imshow(self.x_test[idx])
        plt.axis('off')
        plt.show()

        print(
            f'Категория: {bcolors.BOLD}{bcolors.OKBLUE}{self.__class__.classes[np.argmax(self.y_test[idx])]}{bcolors.ENDC}')
        print()
        print()
        print(f'{bcolors.BOLD}Работа нейронной сети{bcolors.ENDC}')
        print('Тестовое изображение подается на вход нейронной сети')
        print()
        inp = self.model.input
        for i, layer in enumerate(self.model.layers):
            type_layer = layer.__class__.__name__
            outputs = layer.output
            functors = K.function([inp], [outputs])
            layer_outs = functors(self.x_test[idx][None, ...])
            if type_layer == 'Conv2D' or type_layer == 'MaxPooling2D' or type_layer == 'AveragePooling2D' or type_layer == 'Conv2DTranspose':
                cnt_units = layer_outs[0].shape[-1]
                print(
                    f'Значения поступают на {bcolors.OKBLUE}{i + 1}-й слой{bcolors.ENDC} ({type_layer}), содержащий {bcolors.BOLD} {cnt_units} нейронов{bcolors.ENDC}')
                print(f'На выходе слоя получаем набор из {cnt_units} масок (размер: {layer_outs[0].shape[1:-1]}):')
                res = layer_outs[0][0]
                j = 0
                repeat = True
                while repeat:
                    cnt = res.shape[-1] - j * 8
                    if res.shape[-1] - j * 8 > 8:
                        cnt = 8
                    elif res.shape[-1] - j * 8 == 8:
                        cnt = 8
                        repeat = False
                    else:
                        cnt = (res.shape[-1] - j * 8) % 8
                        repeat = False
                    fig, axs = plt.subplots(1, cnt, figsize=(cnt * 2, 1))
                    for i in range(cnt):
                        axs[i].imshow(res[:, :, j * 8 + i], cmap='gray')
                        axs[i].axis('off')
                    plt.show()
                    j += 1
                print()
                print()
            elif type_layer == 'BatchNormalization':
                print(f'Слой BatchNormalization выполняет нормализацию поступивших на вход значений')
                print()
                print()
            elif type_layer == 'Dropout':
                print(f'Слой {bcolors.BOLD} Dropout{bcolors.ENDC}. Используется только при обучении модели')
                print()
            elif type_layer == 'UpSampling2D':
                print(
                    f'Значения поступают на {bcolors.OKBLUE}{i + 1}-й слой{bcolors.ENDC} ({type_layer}), который выполняет расширение изображения')
                cnt_units = layer_outs[0].shape[-1]
                print(f'На выходе слоя получаем набор из {cnt_units} масок (размер: {layer_outs[0].shape[1:-1]}):')
                res = layer_outs[0][0]
                j = 0
                repeat = True
                while repeat:
                    cnt = res.shape[-1] - j * 8
                    if res.shape[-1] - j * 8 > 8:
                        cnt = 8
                    elif res.shape[-1] - j * 8 == 8:
                        cnt = 8
                        repeat = False
                    else:
                        cnt = (res.shape[-1] - j * 8) % 8
                        repeat = False
                    fig, axs = plt.subplots(1, cnt, figsize=(cnt * 2, 1))
                    for i in range(cnt):
                        axs[i].imshow(res[:, :, j * 8 + i], cmap='gray')
                        axs[i].axis('off')
                    plt.show()
                    j += 1
                print()
                print()
            elif type_layer == 'Flatten' or type_layer == 'GlobalMaxPooling2D' or type_layer == 'GlobalAveragePooling2D':
                print(
                    f'Значения поступают на {bcolors.OKBLUE}{i + 1}-й слой{bcolors.ENDC} ({type_layer}), который выполняет развертывание всех данных в один вектор')
                res = layer_outs[0][0]
                print(f'На выходе слоя получаем вектор из {res.shape[0]} значений:')
                weights = layer_outs[0][0]
                if len(weights) > 64:
                    data = [weights[:20]]
                    print(tabulate(data, tablefmt="fancy_grid"))
                    print('................')
                    data = [weights[-20:]]
                    print(tabulate(data, tablefmt="fancy_grid"))
                else:
                    weights = layer_outs[0][0]
                    data = [weights[i:i + 10] for i in range(0, weights.shape[0], 10)]
                    print(tabulate(data, tablefmt="fancy_grid"))
                print()
                print()
            elif type_layer == 'Dense':
                cnt_units = layer.get_weights()[0].shape[1]
                print(
                    f'Значения поступают на {bcolors.OKBLUE}{i + 1}-й слой{bcolors.ENDC} ({type_layer}), содержащий {bcolors.BOLD} {cnt_units} нейронов{bcolors.ENDC}')
                print(f'На выходе слоя получаем вектор из {cnt_units} значений:')
                weights = layer_outs[0][0]
                data = [weights[i:i + 10] for i in range(0, weights.shape[0], 10)]
                print(tabulate(data, tablefmt="fancy_grid"))
                res = layer_outs[0][0]
                print()
                print()
        print(f'{bcolors.BOLD}{bcolors.OKBLUE}Результат работы нейронной сети:{bcolors.ENDC}')
        maxidx = np.argmax(res)
        for i in range(3):
            if i == maxidx:
                print(f'{bcolors.BOLD}{self.__class__.classes[i]}: {round(res[i] * 100, 2)}{bcolors.ENDC}%')
            else:
                print(f'{self.__class__.classes[i]}: {round(res[i] * 100, 2)}%')


worker = Worker()

worker.load_data()

worker.show_samples()

worker.create_sets()

# Вставить создание модели по слоям


# Создание модели
model = Sequential()

# Добавление слоев
model.add(Conv2D(8, (3, 3), input_shape=(108, 192, 3), padding='same', activation='relu'))
model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Nadam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(worker.x_train,
                    worker.y_train,
                    validation_data=(worker.x_test, worker.y_test),
                    batch_size=24,
                    epochs=20,
                    verbose=1)

worker.plot_graphic(history)

worker.layers_demonstration(model)
