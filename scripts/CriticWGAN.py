

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, models
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys



def get_slice(f, id1, id2):
    a = np.vstack((f[:, id1], f[:, id2])).T
    # neg1 = np.min(a[0, :])
    # neg2 = np.min(a[1, :])
    # if neg1 < 1:
    #     a[0, :] += np.abs(neg1) + 1
    # if neg2 < 1:
    #     a[1, :] += np.abs(neg2) + 1
    # log_f = np.log(a.T)
    # print(len(log_f[np.where(log_f == - np.inf)]))
    # print(len(log_f[np.where(log_f[:, 0] == np.inf)]))
    df = pd.DataFrame(a)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).to_numpy()
    return df


def get_discriminator_model(): 
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

    x = layers.Dense(1)(x)
    #x = layers.LeakyReLU(0.2)(x)


    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(100, "tanh")(noise)

    x = layers.Dense(40)(x)
    x = activations.exponential(x)

    x = layers.Dense(IMG_SHAPE)(x)
    x = layers.LeakyReLU(0.2)(x)

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model



class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=1,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

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
            # random_latent_vectors = tf.random.normal(
            #     shape=(batch_size, self.latent_dim)
            # )
            
            # print(fake_images)
            #### lr kleiner; Batch size groesser.
            with tf.GradientTape() as tape:

                # # Generate fake images from the latent vector
                # fake_images = self.generator(random_latent_vectors, training=True)
                ridxs = tf.random.shuffle(idxs)[:batch_size]
                fake_images = tf.gather(pre_sampled, ridxs)

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
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
        # # random_latent_vectors = - tf.ones(shape=(batch_size, self.latent_dim))

        # with tf.GradientTape() as tape:
        #     # Generate fake images using the generator
        #     generated_images = self.generator(random_latent_vectors, training=False)
        #     # Get the discriminator logits for fake images
        #     gen_img_logits = self.discriminator(generated_images, training=False)
        #     # Calculate the generator loss
        #     g_loss = self.g_loss_fn(gen_img_logits)

        # # # # Get the gradients w.r.t the generator loss
        # gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # # Update the weights of the generator using the generator optimizer
        # self.g_optimizer.apply_gradients(
        #     zip(gen_gradient, self.generator.trainable_variables)
        # )
        return {"d_loss": d_loss, "g_loss": 0}



class GANMonitor(keras.callbacks.Callback):
    def __init__(self, train_data, num_img=6, latent_dim=128):
        # self.num_img = num_img
        self.latent_dim = latent_dim
        #self.true = train_data[:, :1000]

    def on_epoch_end(self, epoch, logs):
        
        if epoch % 50 == 0:
            wgan.generator.save(res_dir + "gen.h5") 
            # random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            # gen = self.model.generator(random_latent_vectors).numpy()
            # np.save(res_dir + "gen_output_" + str(epoch) + "_epoch.npy", gen)
            
        if epoch % 1 == 0:
            if os.path.exists(res_dir + "logs.npy"):
              log_file = np.load(res_dir + "logs.npy", allow_pickle = True) 
              log_file = np.append(log_file, np.array(logs["d_loss"]))
              np.save(res_dir + "logs.npy", log_file)
            else:
              np.save(res_dir + "logs.npy", np.array(logs['d_loss']))
              

# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
### plot von gefälschten und echten Daten (disc)
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss #+ 0.5*losses.MSE(y_pred, y_real)


# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def plotPairs(X_train, res_dir, start, end):
    test = DataFrame(X_train)
    for i in range(test.shape[1]):

        if i+1 < test.shape[1]:
            print(i, i+1)

            df12 = test.iloc[:, i:i+2]
            sns_plot = sns.pairplot(df12, diag_kind = 'kde',
                plot_kws = {'alpha': 0.6, 's': 10, 'edgecolor': 'k'},
                height = 4)

            sns_plot.fig.suptitle("Kernel density estimate KDE")
            sns_plot.savefig(res_dir + "INIT_pairplot_" + str(start) + ":" + str(end) + ".png")

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
IMG_SHAPE = 28
BATCH_SIZE = 32000
noise_dim = 28
epochs = 1000
start = 1
end = 3
d_lr = 0.005 #float(sys.argv[1])
g_lr = 0.0001 #float(sys.argv[2])

res_dir = "../exp/disc/trial100/"
os.makedirs(res_dir, exist_ok=True)

X_train = pd.read_csv("true_acc.csv").to_numpy()

gen_fake = tf.keras.models.load_model("../exp/gan/trial5/gen.h5")
random_latent_vectors = tf.random.normal(shape=(140000, 28))
pre_sampled = gen_fake(random_latent_vectors).numpy()
idxs = tf.range(tf.shape(pre_sampled)[0])
del gen_fake

# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5 are recommened
d_model = get_discriminator_model()
d_model.summary()
discriminator_optimizer = keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5, beta_2=0.9)

g_model = get_generator_model()
g_model.summary()
generator_optimizer = keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5, beta_2=0.9)


# Callbacks X_train = numpy
cbk = GANMonitor(train_data=X_train, num_img=5000, latent_dim=noise_dim)
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
    d_loss_fn=discriminator_loss,
)

tf.keras.utils.plot_model(d_model, to_file=res_dir + "discriminator.png", show_shapes=True)
tf.keras.utils.plot_model(g_model, to_file=res_dir + "generator.png", show_shapes=True)
tf.keras.utils.plot_model(wgan, to_file=res_dir + "wgan.png", show_shapes=True)

# Start training
hist = wgan.fit(X_train, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
wgan.discriminator.save(res_dir + "disc.h5")
wgan.generator.save(res_dir + "gen.h5")

