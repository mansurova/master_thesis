import pandas as pd
import tensorflow as tf
import glob, os
import numpy as np

import numbers

import numpy as np
import matplotlib.pyplot as plt


def qqplot(x, y, name, quantiles=None, interpolation='nearest'):

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    np.save(name, np.array([x_quantiles, y_quantiles]))


# Setup
rng = np.random.RandomState(0) 
dim = 1


res_dir = "../exp/gan/trial5/"
if not os.path.exists(res_dir + "10_lines"):
    os.makedirs(res_dir + "10_lines")

x = pd.read_csv("true_acc.csv").to_numpy()#[:, dim]

model = tf.keras.models.load_model(res_dir + "gen.h5")

# Draw quantile-quantile plot
for i in range(28):
    ## save 10 files per dimension
    print("Done: ", i+1)

    for j in range(10):
        random_latent_vectors = tf.random.normal(shape=(5000, 28))
        gen = model(random_latent_vectors).numpy()
        a = x[:, i]
        b = gen[:, i]
        a = np.random.choice(a, b.shape[0])

        qqplot(a, b, name=res_dir + "10_lines/dim_" + str(i) + "_iter_" + str(j) + ".npy", c='r', alpha=0.5, edgecolor='k', ax=None, rug=True)
    

    ## save plots with 10 lines
    plt.figure(1)
    ax = plt.gca()
    files = glob.glob(res_dir + "10_lines/dim_" + str(i) + "_*.npy")

    min_x = 0
    min_y = 0
    max_x = 10
    max_y = 10
    for f in files:
        print(f)
        a = np.load(f)
        min_x = min_x if np.min(a[0, :]) >= min_x else np.min(a[0, :])
        min_y = min_y if np.min(a[1, :]) >= min_y else np.min(a[1, :])
        max_x = max_x if np.max(a[0, :]) <= max_x else np.max(a[0, :])
        max_y = max_y if np.max(a[1, :]) <= max_y else np.max(a[1, :])

        ax.plot(a[0, :], a[1, :], c="black")#, markeredgecolor='none') 
        os.remove(f)
    
    min_v = min_x if min_x < min_y else min_y
    max_v = max_x if max_x < max_y else max_y
    x_axis = np.linspace(min_v, max_v, 10)
    ax.plot(x_axis, x_axis, linestyle='solid', c='gray')

    
    ax.autoscale(enable=True, axis='both', tight=True)   
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(10**(-5), max_v) 
    ax.set_ylim(10**(-5), max_v) 
    plt.xlabel('Ground truth')
    plt.ylabel('Generated')
    plt.gca().set_aspect(1)
    plt.title('Dimension ' + str(i))
    plt.savefig(res_dir + "10_lines/qqplot_" + str(i) + ".png",bbox_inches='tight',pad_inches=0)
    plt.close()

