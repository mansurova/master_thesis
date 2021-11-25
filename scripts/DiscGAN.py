import tensorflow as tf 
import numpy as np 
import keras
from tensorflow.keras import layers, activations, optimizers
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import ModelCheckpoint, TensorBoard


def gen_dataset():
    res_dir = "../exp/raw/trial15/"
    model = tf.keras.models.load_model(res_dir + "gen.h5")

    random_latent_vectors = tf.random.normal(shape=(145081, 28))
    gen = model(random_latent_vectors).numpy()

    print(gen.shape)

    np.save(res_dir + "gen_dataset.npy", gen)


def get_discriminator_model(IMG_SHAPE): 
    img_input = layers.Input(shape=IMG_SHAPE)
    
    # x = layers.Dense(10)(img_input)
    # x = layers.LeakyReLU(0.3)(x)
    # x = layers.Dropout(0.3)(x)

    x = layers.Dense(5)(img_input)  # exponential --> NaN
    x = layers.LeakyReLU(0.3)(x)

    x = layers.Dense(3)(x)
    x = layers.LeakyReLU(0.3)(x) 

    # x = layers.Dense(10)(x)
    # x = layers.LeakyReLU(0.3)(x) 

    x = layers.Dense(1, "sigmoid")(x)
    #x = layers.LeakyReLU(0.2)(x)


    d_model = tf.keras.models.Model(img_input, x, name="discriminator")
    d_model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    d_model.summary()
    return d_model


def load_split_dataset(name):
    true = pd.read_csv("true_acc.csv").to_numpy()
    y = [1] * true.shape[0]
    neg = np.load(name)
    y += [0] * neg.shape[0]
    X = np.vstack((true, neg))

    return train_test_split(
        X, np.array(y), test_size=0.33, random_state=42)

BATCH_SIZE = 2048
res_dir = "../exp/raw/wgan/"
root_dir = "../exp/disc/trial100/"
os.makedirs(root_dir, exist_ok=True)

X_train, X_test, y_train, y_test = load_split_dataset(res_dir + "gen_dataset.npy")

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)



model = get_discriminator_model(IMG_SHAPE=28)

model_checkpoint = ModelCheckpoint(root_dir + 'disc_cb.h5', monitor='val_loss',verbose=1, save_best_only=True)
TB_callback = TensorBoard(log_dir=root_dir + 'logs')

print(X_train.shape[0])
hist = model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=BATCH_SIZE, 
    steps_per_epoch=int(X_train.shape[0]//BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=30000,
    callbacks=[model_checkpoint, TB_callback])

model.save(root_dir + "disc.h5")

