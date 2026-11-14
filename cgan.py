"""Conditional WGAN-GP on CelebA, conditioned on Blond_Hair.

The networks match the course notebook (Foster, Generative Deep Learning,
2nd ed., notebooks/04_gan/03_cgan), which is adapted from Sayak Paul's
Keras conditional-GAN tutorial.

The generator consumes a noise vector concatenated with a 2-class one-hot
vector. The critic consumes the image concatenated with that same label
repeated at every pixel. train_step is where the vector becomes the map.
"""

import tensorflow as tf
from tensorflow.keras import layers, metrics, models, optimizers

IMAGE_SIZE = 64
CHANNELS = 3
CLASSES = 2
Z_DIM = 32
BATCH_SIZE = 128
LEARNING_RATE = 0.00005
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
CRITIC_STEPS = 3
GP_WEIGHT = 10.0
LABEL = "Blond_Hair"
# Course notebook: EPOCHS = 20, then fit(..., epochs=EPOCHS * 100, steps_per_epoch=1).
TRAIN_STEPS = 20 * 100


def expand_label_to_map(one_hot_labels, image_size=IMAGE_SIZE):
    """Broadcast a class vector into the critic's spatial label map.

    Generator label: (batch, CLASSES) one-hot vector.
    Critic label: (batch, image_size, image_size, CLASSES), the same
    one-hot value at every pixel, later concatenated onto the RGB image.

    Passing one_hot_labels straight into the critic is a rank mismatch
    and is the train_step bug this function removes. No new label is
    invented here; each pixel is a copy of the vector the generator saw.
    """
    label_map = one_hot_labels[:, None, None, :]
    label_map = tf.repeat(label_map, repeats=image_size, axis=1)
    label_map = tf.repeat(label_map, repeats=image_size, axis=2)
    return label_map


def build_critic():
    """Critic. Image (64, 64, 3) plus label map (64, 64, 2) -> score."""
    critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    label_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))
    x = layers.Concatenate(axis=-1)([critic_input, label_input])
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
    critic_output = layers.Flatten()(x)
    return models.Model([critic_input, label_input], critic_output, name="critic")


def build_generator():
    """Generator. Noise (Z_DIM,) plus one-hot (CLASSES,) -> 64x64 RGB in [-1, 1]."""
    generator_input = layers.Input(shape=(Z_DIM,))
    label_input = layers.Input(shape=(CLASSES,))
    x = layers.Concatenate(axis=-1)([generator_input, label_input])
    x = layers.Reshape((1, 1, Z_DIM + CLASSES))(x)
    x = layers.Conv2DTranspose(
        128, kernel_size=4, strides=1, padding="valid", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(
        128, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(
        128, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(
        64, kernel_size=4, strides=2, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(0.2)(x)
    generator_output = layers.Conv2DTranspose(
        CHANNELS, kernel_size=4, strides=2, padding="same", activation="tanh"
    )(x)
    return models.Model(
        [generator_input, label_input], generator_output, name="generator"
    )


class ConditionalWGAN(models.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        # The course notebook sets run_eagerly=True so a reader can step
        # through train_step. The arithmetic below is that same step.
        # This run traces it, which is what makes 2,000 updates finish
        # on a CPU residency machine.
        super().compile(run_eagerly=False)
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric = metrics.Mean(name="c_gp")
        self.c_loss_metric = metrics.Mean(name="c_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        # Course notebook draws alpha from a normal. The WGAN-GP paper
        # uses Uniform(0, 1). Kept as the notebook writes it.
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        # Model.fit resets these at each epoch. This run calls train_step
        # directly, so reset here and the logged value is this update.
        self.c_loss_metric.reset_state()
        self.c_wass_loss_metric.reset_state()
        self.c_gp_metric.reset_state()
        self.g_loss_metric.reset_state()

        real_images, one_hot_labels = data

        # --- required change -------------------------------------------------
        # one_hot_labels is the generator format: (batch, 2).
        # The critic's second input is a map: (batch, 64, 64, 2).
        # Expand first, then use the map for real, fake, and the penalty.
        image_one_hot_labels = expand_label_to_map(one_hot_labels, IMAGE_SIZE)
        # --------------------------------------------------------------------

        batch_size = tf.shape(real_images)[0]

        for _ in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    [random_latent_vectors, one_hot_labels], training=True
                )

                # The generator sees the vector. The critic sees the map.
                fake_predictions = self.critic(
                    [fake_images, image_one_hot_labels], training=True
                )
                real_predictions = self.critic(
                    [real_images, image_one_hot_labels], training=True
                )

                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(
                    real_predictions
                )
                c_gp = self.gradient_penalty(
                    batch_size, real_images, fake_images, image_one_hot_labels
                )
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images = self.generator(
                [random_latent_vectors, one_hot_labels], training=True
            )
            fake_predictions = self.critic(
                [fake_images, image_one_hot_labels], training=True
            )
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


def build_cgan():
    critic = build_critic()
    generator = build_generator()
    cgan = ConditionalWGAN(
        critic=critic,
        generator=generator,
        latent_dim=Z_DIM,
        critic_steps=CRITIC_STEPS,
        gp_weight=GP_WEIGHT,
    )
    optimizer = lambda: optimizers.Adam(  # noqa: E731
        learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2
    )
    cgan.compile(c_optimizer=optimizer(), g_optimizer=optimizer())
    return cgan


def one_hot(class_index, batch):
    """class_index 0 is not blond [1, 0]; class_index 1 is blond [0, 1]."""
    vector = [1.0, 0.0] if class_index == 0 else [0.0, 1.0]
    return tf.constant([vector] * batch, dtype=tf.float32)
