import seaborn as sns
import random
import numpy as np
import tensorflow as tf
import intensivedayone


# Установочные параметры

sns.set(style='darkgrid')
seed_value = 12
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# Загрузка модуля
# gdown.download('https://storage.yandexcloud.net/aiueducation/Intensive/intensivedayone.py', None, quiet=True)


demo_ai = intensivedayone.TerraIntensive()

demo_ai.load_dataset('Молочная_продукция')

demo_ai.samples()

demo_ai.create_sets()

# Задание слоев нейронной сети
layers = 'Сверточный2Д-64-3\
 Сверточный2Д-64-3\
 Выравнивающий\
 Полносвязный-64\
 Полносвязный-3'

# Создание нейронной сети
demo_ai.create_model(layers)

# Обучение нейронной сети
demo_ai.train_model(epochs=20)