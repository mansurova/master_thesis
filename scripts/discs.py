import pandas as pd
import tensorflow as tf
import glob 
import numpy as np

import numbers

import numpy as np
import matplotlib.pyplot as plt


def inference_wgan():
    res_dir = "../exp/raw/wgan/"
    # res_dir = "../exp/raw/trial15/"
    fraud = pd.read_csv("fraud_acc.csv").to_numpy()
    true = pd.read_csv("true_acc.csv").to_numpy()
    gen = np.load(res_dir + "gen_dataset.npy")
    # gen = np.load(res_dir + "gen_output_950_epoch.npy")

    model = tf.keras.models.load_model(res_dir + "disc.h5")

    result = model(fraud)
    np.save("../results/wgan_fraud.npy", result)

    result = model(true)
    np.save("../results/wgan_true.npy", result)

    result = model(gen)
    np.save("../results/wgan_gen.npy", result)

    print(result.shape)


def inference_gan():
    res_dir = "../exp/gan/trial5/"
    fraud = pd.read_csv("fraud_acc.csv").to_numpy()
    true = pd.read_csv("true_acc.csv").to_numpy()
    gen = np.load(res_dir + "gan_generated.npy")

    model = tf.keras.models.load_model(res_dir + "disc.h5")

    result = model(fraud)
    np.save("../results/gan_fraud.npy", result)

    result = model(true)
    np.save("../results/gan_true.npy", result)

    result = model(gen)
    np.save("../results/gan_gen.npy", result)

    print(result.shape)


def inference_clf():
    res_dir = "../exp/disc/DiscGAN/"
    fraud = pd.read_csv("fraud_acc.csv").to_numpy()
    true = pd.read_csv("true_acc.csv").to_numpy()
    #gen = np.load(res_dir + "gen_output_2800_epoch.npy")

    model = tf.keras.models.load_model(res_dir + "disc.h5")

    result = model(fraud)
    np.save("../results/DiscGAN_fraud.npy", result)

    result = model(true)
    np.save("../results/DiscGAN_true.npy", result)


    res_dir = "../exp/disc/CriticWGAN/"
    fraud = pd.read_csv("fraud_acc.csv").to_numpy()
    true = pd.read_csv("true_acc.csv").to_numpy()
    #gen = np.load(res_dir + "gen_output_2800_epoch.npy")

    model = tf.keras.models.load_model(res_dir + "disc.h5")

    result = model(fraud)
    np.save("../results/CriticWGAN_fraud.npy", result)

    result = model(true)
    np.save("../results/CriticWGAN_true.npy", result)

    # result = model(gen)
    # np.save("../exp/results/gan_gen.npy", result)

    print(result.shape)


def plot_hists():

    ####
    wgan_true = np.squeeze(np.load("../results/CriticWGAN_true.npy"))
    wgan_fraud = np.squeeze(np.load("../results/CriticWGAN_fraud.npy"))

    plt.figure()
    plt.title("CriticWGAN")
    ax = plt.gca()
    n, bins, patches = plt.hist(wgan_fraud, 100, density=True, facecolor='blue', alpha=0.5, label="CriticWGAN_fraud")
    n, bins, patches = plt.hist(wgan_true, 100, density=True, facecolor='red', alpha=0.5, label="CriticWGAN_true")
    plt.legend()
    plt.xlabel("location")
    plt.ylabel("#samples")
    plt.show()

    ####
    wgan_fraud = np.squeeze(np.load("../results/wgan_fraud.npy"))
    wgan_true = np.squeeze(np.load("../results/wgan_true.npy"))
    wgan_gen = np.squeeze(np.load("../results/wgan_gen.npy"))

    wgan_fraud = np.log(wgan_fraud)
    wgan_true = np.log(wgan_true)
    wgan_gen = np.log(wgan_gen)


    plt.figure()
    plt.title("Wgan")
    ax = plt.gca()
    n, bins, patches = plt.hist(wgan_fraud, 100, density=True, facecolor='blue', alpha=0.5, label="wgan_fraud")
    n, bins, patches = plt.hist(wgan_true, 100, density=True, facecolor='red', alpha=0.5, label="wgan_true")
    n, bins, patches = plt.hist(wgan_gen, 100, density=True, facecolor='green', alpha=0.5, label="wgan_gen")
    plt.xlabel("location")
    plt.ylabel("#samples")
    plt.legend()
    plt.show()

    ####
    gan_true = np.squeeze(np.load("../results/gan_true.npy"))
    gan_fraud = np.squeeze(np.load("../results/gan_fraud.npy"))
    gan_gen = np.squeeze(np.load("../results/gan_gen.npy"))

    plt.figure()
    plt.title("GAN")
    ax = plt.gca()
    n, bins, patches = plt.hist(gan_fraud, 100, density=True, facecolor='blue', alpha=0.5, label="gan_fraud")
    n, bins, patches = plt.hist(gan_true, 100, density=True, facecolor='red', alpha=0.5, label="gan_true")
    n, bins, patches = plt.hist(gan_gen, 100, density=True, facecolor='green', alpha=0.5, label="gan_gen")
    plt.legend()
    plt.xlabel("location")
    plt.ylabel("#samples")
    plt.show()


    ####
    gan_true = np.squeeze(np.load("../results/DiscGAN_true.npy"))
    gan_fraud = np.squeeze(np.load("../results/DiscGAN_fraud.npy"))

    plt.figure()
    plt.title("DiscGAN")
    ax = plt.gca()
    n, bins, patches = plt.hist(gan_fraud, 100, density=True, facecolor='blue', alpha=0.5, label="DiscGAN_fraud")
    n, bins, patches = plt.hist(gan_true, 100, density=True, facecolor='red', alpha=0.5, label="DiscGAN_true")
    plt.legend()
    plt.xlabel("location")
    plt.ylabel("#samples")
    plt.show()

    

# inference_gan()
# inference_wgan()
# inference_clf()
plot_hists()