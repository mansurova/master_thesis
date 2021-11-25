import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import os


def gen_dataset1(NUM_DATPOINTS, IMG_SHAPE, toLoad=False):
    if toLoad:
        return np.load("../data/2D_100000data.npy")
    else:
        mean = [0, 0]
        cov = [[1, -0.8], [-0.8, 1]]  # diagonal covariance
        Z = np.random.multivariate_normal(mean, cov, NUM_DATPOINTS) #[:, :, np.newaxis]
        print(Z.shape)
        return Z.astype("float32")

def gen_dataset2(NUM_DATPOINTS, IMG_SHAPE, toLoad=False):
    if toLoad:
        return np.load("../data/2D_100000data.npy")
    else:
        np.random.seed(0)
        sigma = 0.5
        # generate spherical data centered on (20, 20)
        shifted_gaussian = sigma * np.random.randn(NUM_DATPOINTS//2, 2) + np.array([1, 1.5])
        
        # generate zero centered stretched Gaussian data
        C = np.array([[0., -0.2], [1.5, .7]])
        stretched_gaussian = np.dot(np.random.randn(NUM_DATPOINTS//2, 2), C)

        Z = np.vstack([shifted_gaussian, stretched_gaussian])#[:, :, np.newaxis]
        
        return np.exp(Z.astype("float32"))


def get_slice(f, id1, id2, to_log=False):
    a = np.vstack((f[:, id1], f[:, id2])).T
    if to_log:
        if np.min(a[:, 0]) < 1:
            a[:, 0] += np.abs(np.min(a[:, 0])) + 1
        if np.min(a[:, 1]) < 1:
            a[:, 1] += np.abs(np.min(a[:, 1])) + 1
        log_a = np.log(a) # real
    else:
        log_a = a
    da = pd.DataFrame(log_a)
    return da.replace([np.inf, -np.inf], np.nan).dropna(axis=0).to_numpy()


def get_discriminator_model():
    img_input = layers.Input(shape=IMG_SHAPE)

    x = layers.Dense(5)(img_input)  # exponential --> NaN
    x = layers.LeakyReLU(0.3)(x)

    x = layers.Dense(3)(x)
    x = layers.LeakyReLU(0.3)(x) 

    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))

    # x = layers.Dense(5)(noise)
    # x = layers.LeakyReLU(0.3)(x) 
    # x = layers.Dense(IMG_SHAPE)(x)
    # x = layers.LeakyReLU(0.3)(x) 

    x = layers.Dense(50, "tanh")(noise)

    x = layers.Dense(20)(x)
    x = activations.exponential(x)

    x = layers.Dense(IMG_SHAPE)(x)
    ##x = layers.LeakyReLU(0.2)(x)

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, run_eagerly=False):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.run_eagerly = run_eagerly

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):

        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            # print(fake_images)
            with tf.GradientTape() as tape:

                # # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                
                # fake_images = tf.random.normal(
                #     shape=(batch_size, IMG_SHAPE)
                # )

                # # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)

                # Get the logits for real images
                real_logits = self.discriminator(real_images, training=True)

                # fake_images = trans.inverse_transform(fake_images)        

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}



class GANMonitor(keras.callbacks.Callback):
    def __init__(self, train_data, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.train_data = train_data #[NUM_DATPOINTS//2 - self.num_img//2 : NUM_DATPOINTS//2 + self.num_img//2, :]
    def on_epoch_end(self, epoch, logs):
        if epoch % 1 == 0:
            if os.path.exists(res_dir + "logs.npy"):
              log_file = np.load(res_dir + "logs.npy", allow_pickle = True) 
              log_file = np.append(log_file, np.array([[logs["d_loss"], logs["g_loss"]]]), axis=0)
              np.save(res_dir + "logs.npy", log_file)
            else:
              np.save(res_dir + "logs.npy", np.array([[logs["d_loss"], logs["g_loss"]]]))
        
        if epoch % 10 == 0:
            wgan.discriminator.save(res_dir + "disc.h5")
            wgan.generator.save(res_dir + "gen.h5") 
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            gen_output = self.model.generator(random_latent_vectors).numpy()
            np.save(res_dir + "wgan_pairs_%d.npy" % epoch, gen_output)
        # if epoch % 550 == 0:
        #     y = ["generated"] * self.num_img
        #     set1 = DataFrame(np.squeeze(gen_output))
        #     set1["type"] = y
        #     y = ["train"] * self.num_img
        #     set2 = DataFrame(self.train_data) 
        #     set2["type"] = y
        #     df12 = set2.append(set1)
        #     sns_plot = sns.pairplot(df12, hue = 'type', diag_kind = 'kde',
        #         plot_kws = {'alpha': 0.2, 's': 10, 'edgecolor': 'k'},
        #         height = 4)
        #     l1 = "{:.3f}".format(-logs['d_loss'])
        #     l2 = "{:.3f}".format(logs['g_loss'])
        #     sns_plot.fig.suptitle("G_loss = " + l2 + ", W-dist = " + l1, y=1.08)
        #     sns_plot.savefig(res_dir + "wgan_pairs_trans_%d.png" % epoch)


# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def plotPairs(X_train, res_dir, start, end, trans=True):
    test = DataFrame(X_train)
    for i in range(test.shape[1]):

        if i+1 < test.shape[1]:
            print(i, i+1)

            df12 = test.iloc[:, i:i+2]
            sns_plot = sns.pairplot(df12, diag_kind = 'kde',
                plot_kws = {'alpha': 0.6, 's': 10, 'edgecolor': 'k'},
                height = 4)

            sns_plot.fig.suptitle("Kernel density estimate KDE")
            if trans:
                sns_plot.savefig(res_dir + "INIT_pairplot_trans_" + str(start) + ":" + str(end) + ".png")
            else:
                sns_plot.savefig(res_dir + "INIT_pairplot_real_" + str(start) + ":" + str(end) + ".png")

            # plt.show()


def save_history(history):
    np.save(res_dir + "loss.npy", history.history)
    # Plot history: Wasserstein distance
    plt.plot(history.history['g_loss'], label='generator')
    plt.plot(history.history['d_loss'], label='discriminator')
    plt.title('Wasserstein loss')
    plt.ylabel('Loss')
    plt.xlabel('No. epoch')
    plt.legend()
    plt.savefig(res_dir + "wgan_history.png")


############################################################################################################
##########################################___ MAIN ___######################################################
############################################################################################################
IMG_SHAPE = 2
BATCH_SIZE = 2048
noise_dim = 4
epochs = 3000
start = 0
end = 20
NUM_DATPOINTS = 5000
res_dir = "../exp/raw/trial100/" #"../exp/raw/trial3/"
os.makedirs(res_dir, exist_ok=True)
isTrans = True

# Load data
loadData = False
# X_train = gen_dataset2(NUM_DATPOINTS, IMG_SHAPE, loadData) # GMM
f = pd.read_csv("true_acc.csv").to_numpy()

# X_train = get_slice(f, start, end, to_log=isTrans)
f_trans = get_slice(f, start, end, to_log=False)

# trans = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
# f_trans = trans.fit_transform(f_trans)

# plotPairs(X_train, res_dir, start, end, trans=isTrans)
plotPairs(f_trans, res_dir, start, end, trans=False)

X_train = f_trans


# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5 are recommened
d_model = get_discriminator_model()
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.02, beta_1=0.5, beta_2=0.9)

g_model = get_generator_model()
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)


# Callbacks X_train = numpy
cbk = GANMonitor(train_data=X_train, num_img=X_train.shape[0], latent_dim=noise_dim) #X_train.shape[0]
# Get the wgan model
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=noise_dim,
    discriminator_extra_steps=3,
)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=res_dir,
    save_weights_only=True,
    monitor='d_loss',
    mode='min',
    save_best_only=True)

# Compile the wgan model
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss#,
    # run_eagerly=True
)


tf.keras.utils.plot_model(d_model, to_file=res_dir + "discriminator.png", show_shapes=True)
tf.keras.utils.plot_model(g_model, to_file=res_dir + "generator.png", show_shapes=True)
tf.keras.utils.plot_model(wgan, to_file=res_dir + "wgan.png", show_shapes=True)

# Start training
hist = wgan.fit(X_train, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
wgan.discriminator.save(res_dir + "disc.h5")
wgan.generator.save(res_dir + "gen.h5")
