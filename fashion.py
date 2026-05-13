import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

# Load dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Show dataset samples
plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')
plt.suptitle("Fashion MNIST Samples")
plt.show()

# Normalize to [-1,1]
X_train = X_train / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

print("Data range:", X_train.min(), X_train.max())

img_shape = (28, 28, 1)
latent_dim = 100

# Generator
def build_generator():
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(256),
        LeakyReLU(negative_slope=0.2),

        Dense(512),
        LeakyReLU(negative_slope=0.2),

        Dense(2048),
        LeakyReLU(negative_slope=0.2),

        Dense(28 * 28 * 1, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = Sequential([
        Input(shape=img_shape),
        Flatten(),

        Dense(512),
        LeakyReLU(negative_slope=0.2),

        Dense(256),
        LeakyReLU(negative_slope=0.2),

        Dense(1, activation='sigmoid')
    ])
    return model

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator (lower LR)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0001, 0.5),
    metrics=['accuracy']
)

# GAN setup
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5)
)

# Generate images
def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r*c, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c, figsize=(6,6))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    plt.suptitle(f"Generated Images - Epoch {epoch}")
    plt.show()

# Training
def train(epochs, batch_size=128, sample_interval=500):
    half_batch = batch_size // 2

    for epoch in range(epochs):

        # Train Discriminator
        discriminator.trainable = True

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)

        # Label smoothing
        real_y = np.ones((half_batch, 1)) * 0.9
        fake_y = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        discriminator.trainable = False

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_y)

        # Print progress
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

        # Show generated images
        if epoch % sample_interval == 0:
            sample_images(epoch)

# Train model
train(epochs=8000, batch_size=128, sample_interval=500)
