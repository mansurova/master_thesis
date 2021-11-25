from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, activations, optimizers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GAN():
    def __init__(self, res_dir):
        self.IMG_SHAPE = 28
        self.latent_dim = 28 #100
        self.res_dir = res_dir
        optimizer = optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])


        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



    def build_generator(self):

        noise = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(100, "tanh")(noise)

        x = layers.Dense(40)(x)
        x = activations.exponential(x)

        x = layers.Dense(self.IMG_SHAPE)(x)
        x = layers.LeakyReLU(0.2)(x)

        g_model = models.Model(noise, x, name="generator")
        return g_model


    def build_discriminator(self):

        img_input = layers.Input(shape=self.IMG_SHAPE)
    
        x = layers.Dense(5)(img_input)  
        x = layers.LeakyReLU(0.3)(x)

        x = layers.Dense(3)(x)
        x = layers.LeakyReLU(0.3)(x) 

        x = layers.Dense(1, "sigmoid")(x)

        d_model = models.Model(img_input, x, name="discriminator")
        return d_model


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = pd.read_csv("true_acc.csv").to_numpy()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
                # Select a random batch of images
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                # imgs = X_train[idx]
            for step in range(int(X_train.shape[0] // batch_size)):

                
                imgs = X_train[step*batch_size:step*batch_size + batch_size]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

        
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            logs = np.load(self.res_dir + "logs.npy")
            logs = np.vstack([logs, [[d_loss[0], g_loss, 100*d_loss[1] ]]])
            np.save(self.res_dir + "logs.npy", logs)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 500 == 0:
                self.discriminator.save(res_dir + "disc.h5")
                self.generator.save(res_dir + "gen.h5")

    def sample_images(self, epoch):
        num_img = 5000
        noise = np.random.normal(0, 1, (num_img, self.latent_dim))
        gen = self.generator.predict(noise)

        np.save(self.res_dir + "gen_output_" + str(epoch) + "_epoch.npy", gen)


if __name__ == '__main__':
    res_dir = "trial5/"
    os.makedirs(res_dir, exist_ok=True)

    np.save(res_dir + "logs.npy", np.array([[0, 0, 0]]))
    gan = GAN(res_dir)
    #gan.discriminator.load_weights("../exp/disc/trial1/disc.h5")

    gan.train(epochs=10000, batch_size=32000, sample_interval=200)

    gan.generator.save(res_dir + "gen.h5")
    gan.discriminator.save(res_dir + "disc.h5")
