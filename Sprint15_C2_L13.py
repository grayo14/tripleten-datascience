# importar librerías

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np


# función para cargar datos

def load_data():
  (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
  features_train = features_train.reshape(features_train.shape[0], -1) / 255
  features_test = features_test.reshape(features_test.shape[0], -1) / 255

  return (features_train, target_train),(features_test, target_test)


# función para crear modelo

def create_model(units,features_train,activation,loss,optimizer,metrics):
  model = keras.models.Sequential()

  model.add(keras.layers.Dense(units=units,input_dim=features_train.shape[1],activation=activation))

  model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
  return model

# función para entrenar modelo

def train_model(model,features_train,target_train,features_test,target_test):
  model.fit(
    features_train,
    target_train,
    validation_data=(features_test,target_test),
    verbose=2)


# probar funciones

(features_train, target_train),(features_test, target_test) = load_data()

model_s = create_model(
    units=10,
    features_train=features_train,
    activation='softmax',
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['acc']
    )

train_model(
    model=model_s,
    features_train=features_train,
    target_train=target_train,
    features_test=features_test,
    target_test=target_test
    )

# validar tamaños y escalado de los features

print(features_train.shape)
print(features_train[0,50:150])