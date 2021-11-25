import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import glob 
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf


def gen_dataset2(NUM_DATPOINTS, toLoad=False):
    if toLoad:
        return np.load("../data/2D_100000data.npy")
    else:
        np.random.seed(0)
        sigma = 0.5
        # generate spherical data centered on (20, 20)
        shifted_gaussian = sigma * np.random.randn(NUM_DATPOINTS//2, 2) + np.array([1, 1.5])
        print(shifted_gaussian.shape)
        # generate zero centered stretched Gaussian data
        C = np.array([[0., -0.2], [1.5, .7]])
        stretched_gaussian = np.dot(np.random.randn(NUM_DATPOINTS//2, 2), C)
        print(np.std(shifted_gaussian, axis = 0))
        Z = np.vstack([shifted_gaussian, stretched_gaussian])#[:, :, np.newaxis]
        plt.title("Generated GMM dataset")
        plt.scatter(stretched_gaussian[:, 0], stretched_gaussian[:, 1], s=1, label="stretched, zero centered")
        plt.scatter(shifted_gaussian[:, 0], shifted_gaussian[:, 1], s=1, label="shifted, symmetric covariance")
        plt.legend()
        plt.show()

        return Z.astype("float32")


def p_pairplots(gen_output):
    
    #gen_output = np.load(res_dir + name)[:5000, dimension]

    df12 = DataFrame(gen_output)
    sns_plot = sns.pairplot(df12, diag_kind = 'kde',
        plot_kws = {'alpha': 0.2, 's': 10, 'edgecolor': 'k'},
        height = 2)
    plt.show()
    #sns_plot.savefig(res_dir + name[:-3] + "png")



def save_history(history, path):
    #np.save(res_dir + "training_info.npy", history.history)
    # Plot history: Wasserstein distance
    plt.figure(figsize=(7, 5))

    #plt.plot(history[:, 1], label='gen loss')
    # plt.plot(history[:, 0], label='critic loss')
    if len(history.shape) > 1:
        plt.plot(history[:, 0], label='critic loss')
    else:
        plt.plot(history, label='critic loss')

    # plt.plot(history[:, 2]/100, label='disc acc')
    #plt.plot(history, label='critic loss')

    plt.title('Trainingsprozess')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    # plt.ylim(-1500000, 1000)
    plt.legend()
    plt.show()
    plt.savefig(path + "wgan_history.png")
       

def plot_pairplots(train_data, gen_output, res_dir, name, dimension=None):

    y = ["generated"] * gen_output.shape[0]
    set1 = DataFrame(np.squeeze(gen_output))
    set1["type"] = y
    y = ["train"] * train_data.shape[0]
    set2 = DataFrame(train_data) 
    set2["type"] = y

    df12 = set2.append(set1)
    g = sns.pairplot(df12, hue = 'type', diag_kind = 'kde',
        plot_kws = {'alpha': 0.2, 's': 10, 'edgecolor': 'k'},
        height = 3)

    plt.show()
    #sns_plot.savefig(res_dir + name[:-3] + "png")


def get_slice(true, gen, id1, id2):

    a = np.vstack((true[:, id1], true[:, id2])).T
    b = np.vstack((gen[:, id1], gen[:, id2])).T

    da = pd.DataFrame(a)
    da = da.replace([np.inf, -np.inf], np.nan).dropna(axis=0)#.to_numpy()
    db = pd.DataFrame(b)
    db = db.replace([np.inf, -np.inf], np.nan).dropna(axis=0)#.to_numpy()

    return da, db


def gan_save_history(history, path):
    #np.save(res_dir + "training_info.npy", history.history)
    # Plot history: Wasserstein distance
    plt.figure(figsize=(20, 10))

    plt.plot(history[:, 0], label='disc loss')
    plt.plot(history[:, 1], label='gen loss')
    plt.plot(history[:, 2]/100, label='disc acc')

    # plt.plot(history, label='disc loss')

    plt.title('Trainingsprozess')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    # plt.ylim(0, 2)
    plt.legend()
    plt.show()
    # plt.savefig(path + "wgan_history.png")

#gen_dataset2(50000)


### unpack generated points 2D
start = 7
end = 22
IMG_LEN = 140000
res_dir = "../exp/gan/trial5/"
save_history(np.load(res_dir + "logs.npy"), res_dir)
# files = glob.glob(res_dir + "gen_output_950_epoch.npy")[0]
# #print(files)
# gan_save_history(np.load(res_dir + "logs.npy"), res_dir)


f = pd.read_csv("true_acc.csv").to_numpy()[:IMG_LEN, :]
gen_fake = tf.keras.models.load_model(res_dir + "gen.h5")
random_latent_vectors = tf.random.normal(shape=(IMG_LEN, 28))
gen_output = gen_fake(random_latent_vectors).numpy()
# # gen_output = np.load(files)[:IMG_LEN, :]

# trans = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
# f = trans.fit_transform(f)
# gen_output = trans.fit_transform(gen_output)

f = np.log(f)
gen_output = np.log(gen_output)

da, a= get_slice(f, f, start, end)
a, db= get_slice(gen_output, gen_output, start, end) 
plot_pairplots(da, db, res_dir, None)


