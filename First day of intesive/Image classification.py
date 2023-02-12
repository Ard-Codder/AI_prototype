import seaborn as sns
import random
import numpy as np
import tensorflow as tf
import intensivedayone
import importlib as im

# Установочные параметры
sns.set(style='darkgrid')
seed_value = 12
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

im.reload(intensivedayone)

demo_ai = intensivedayone.TerraIntensive()

demo_ai.load_dataset('авто')

demo_ai.samples()

demo_ai.create_sets()

# Создание слоев нейронной сети
layers = 'conv2d-128-10\
    conv2d-64-5\
    leveling\
    fullconn-64\
    fullconn-3'

# Создание нейронной сети
demo_ai.create_model(layers)

# Проверка нейронной сети
# demo_ai.test_model()

# Обучение нейронной сети
demo_ai.train_model(epochs=50)
