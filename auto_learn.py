import sys
import tensorflow as tf
import tensorflow_probability as tfp
import os
from tensorflow.keras import layers
from datetime import datetime
import argparse
import scipy.stats

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    print(f'GPUs found: {physical_devices}')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('No GPUs found')

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument('--weights', type=str, default='', help='Path to weights, for example "logs/20220104-213105/weights.160".')
parser.add_argument('--img-path', type=str, default='img_align_celeba', help='Folder in which the images are located')

args = parser.parse_args()

# Initial learning rate
lr = 0.1
img_size = 64
batch_size = 32
# How many frames to take from the dataset for training. By default, take all
n_items = sys.maxsize
epochs = 1000
# Dimension of the noise vector of the decoder
code_dim = 128
# How many examples to generate for visualization during training
num_examples_to_generate = 16
# The higher, the more the training aims to produces a noise vector in the decoder which is shaped like a normal distribution
regularization_multiplier = 0.1
corr_regularization_multiplier = 1.

split = 0.5

def make_decoder_model():
  dec = tf.keras.Sequential(
    [
      tf.keras.Input(shape=(code_dim,)),
      layers.Reshape ((1, 1, code_dim)),
      layers.Conv2DTranspose(256, kernel_size=4, strides=4, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2DTranspose(3, kernel_size=4, strides=2, activation='sigmoid', padding='same')
    ],
    name="decoder",
  )

  return dec

def make_encoder_model():
  enc = tf.keras.Sequential(
    [
      layers.Input(shape=(img_size, img_size, 3)),
      layers.Conv2D(32, (4, 4), padding='same', strides=2),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(64, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False),
      layers.BatchNormalization(),
      layers.LeakyReLU(alpha=0.2),
      layers.Conv2D(code_dim, (4, 4), padding='same', strides=4),
      layers.Flatten(),
    ],
    name="encoder",
  )

  return enc


class CustomModel(tf.keras.Model):

  def __init__(self):
    super(CustomModel, self).__init__()

    self.seed = tf.random.normal([batch_size, code_dim])
    self.quantiles = tf.tile(tf.reshape(tf.constant([scipy.stats.norm.ppf(item) for item in tf.cast(tf.linspace(0, 1, int(batch_size+2))[1:-1], tf.float32).numpy().tolist()], dtype=tf.float32), (batch_size, 1)), (1, code_dim))

    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.regularization_loss_tracker = tf.keras.metrics.Mean(name="regularization_loss")
    self.regularization_corr_tracker = tf.keras.metrics.Mean(name="regularization_corr")
    self.regularization_corr_loss_tracker = tf.keras.metrics.Mean(name="regularization_corr_loss")
    self.mean_tracker = tf.keras.metrics.Mean(name="mean")
    self.std_tracker = tf.keras.metrics.Mean(name="std")
    self.skew_tracker = tf.keras.metrics.Mean(name="skew")
    self.kurt_tracker = tf.keras.metrics.Mean(name="kurt")

    ones = tf.ones((code_dim, code_dim))
    mask_a = tf.linalg.band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    self.mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask

    self.decoder = make_decoder_model()
    self.decoder.summary()
    self.encoder = make_encoder_model()
    self.encoder.summary()

  @tf.function
  def call(self, _, training=False):
    return self.inference()

  def generate(self, noise):
    img = self.decoder(noise, training=False)
    return img

  @tf.function
  def inference(self):
    generated_images = self.decoder(self.seed, training=False)
    return generated_images

  @tf.function
  def train_step(self, images):

    with tf.GradientTape() as tape:

      code = self.encoder(images)

      sorted_code = tf.sort(code, axis=0)
      regularization_loss = tf.reduce_mean((sorted_code-self.quantiles)**2.)

      correlations = tfp.stats.correlation(code)

      corr = correlations[self.mask]**2
      correlation_loss = tf.reduce_mean(corr)

      rec = self.decoder(code)
      loss = tf.reduce_mean((images - rec)**2.)

      total_loss = loss + regularization_multiplier*regularization_loss + corr_regularization_multiplier*correlation_loss

    gradients = tape.gradient(total_loss, 
      self.encoder.trainable_variables + self.decoder.trainable_variables)

    self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))

    mean = tf.reduce_mean(code)
    std = tf.math.reduce_std(code)
    skew = tf.reduce_mean((code - mean)**3)/std**3
    kurt = tf.reduce_mean((code - mean)**4)/std**4

    self.loss_tracker.update_state(loss)
    self.regularization_loss_tracker.update_state(regularization_loss)
    self.regularization_corr_loss_tracker.update_state(correlation_loss)
    self.regularization_corr_tracker.update_state(tf.reduce_sum(tf.sqrt(corr))/tf.reduce_sum(tf.cast(self.mask, tf.float32)))
    self.mean_tracker.update_state(mean)
    self.std_tracker.update_state(std)
    self.skew_tracker.update_state(skew)
    self.kurt_tracker.update_state(kurt)

    return {
      "loss": self.loss_tracker.result(),
      "regularization_loss": self.regularization_loss_tracker.result(),
      "regularization_corr": self.regularization_corr_tracker.result(),
      "regularization_corr_loss": self.regularization_corr_loss_tracker.result(),
      "mean": self.mean_tracker.result(),
      "std": self.std_tracker.result(),
      "skew": self.skew_tracker.result(),
      "kurt": self.kurt_tracker.result()}

class CustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs):
    if epoch == 0:
      self.model.lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(lr, epochs*batches_per_epoch)

      self.model.optimizer = tf.keras.optimizers.SGD(self.model.lr_decayed_fn)
      self.model.val_data = next(val_ds.__iter__())

      with file_writer.as_default():
        tf.summary.image("In imgs", self.model.val_data, step=epoch, max_outputs=num_examples_to_generate)
        
    with file_writer.as_default():
      tf.summary.scalar('lr', self.model.lr_decayed_fn(epoch*batches_per_epoch), step=epoch)
    
    self.model.loss_tracker.reset_states()
    self.model.regularization_loss_tracker.reset_states()
    self.model.regularization_corr_loss_tracker.reset_states()
    self.model.regularization_corr_tracker.reset_states()
    self.model.mean_tracker.reset_states()
    self.model.std_tracker.reset_states()
    self.model.skew_tracker.reset_states()
    self.model.kurt_tracker.reset_states()

  def on_epoch_end(self, epoch, logs=None):
    generated_images = self.model.inference()
    codes = self.model.encoder(self.model.val_data, training=False)
    rec = self.model.decoder(codes, training=False)

    with file_writer.as_default():
      tf.summary.image("Made up imgs", generated_images, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.image("Out imgs", rec, step=epoch, max_outputs=num_examples_to_generate)
      tf.summary.histogram("Out noise", tf.reshape(codes, (-1,)), step=epoch)
      split_codes = tf.unstack(codes, axis=-1)

      for i, c in enumerate(split_codes):
        tf.summary.histogram(f"Out noise d{i}", tf.reshape(c, (-1,)), step=epoch)

      for key in logs:
        tf.summary.scalar(key, logs[key], step=epoch)

model = CustomModel()
model.compile()#run_eagerly=True)
if args.weights != '':
  
  print("Loading weights from", args.weights)
  model.load_weights(args.weights)

train_ds = tf.keras.utils.image_dataset_from_directory(
  args.img_path,
  validation_split=0.1,
  subset="training",
  seed=123,
  shuffle=True,
  image_size=(img_size, img_size),
  batch_size=None,
  label_mode=None)

train_ds = train_ds.batch(batch_size, drop_remainder=True).map(lambda x: x / 255.0).prefetch(10)

val_ds = tf.keras.utils.image_dataset_from_directory(
  args.img_path,
  validation_split=0.1,
  subset="validation",
  seed=123,
  shuffle=True,
  image_size=(img_size, img_size),
  batch_size=None,
  label_mode=None)

val_ds = val_ds.batch(batch_size, drop_remainder=True).map(lambda x: x / 255.0).prefetch(10)

batches_per_epoch = len(train_ds)

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
model.fit(x=train_ds,
          # validation_data=val_ds,
          epochs=epochs,
          # shuffle=True,
          callbacks=[
            CustomCallback(), 
            tf.keras.callbacks.ModelCheckpoint(
              os.path.join(logdir, "weights.{epoch:02d}"), verbose=1, save_weights_only=True, save_best_only=False, save_freq=10*len(train_ds))
          ])
