import tensorflow as tf
from keras import layers, models, optimizers, Sequential
import numpy as np
import matplotlib.pyplot as plt
import os
import time

class CycleGAN:
    def __init__(self, img_shape=(128, 128, 3), gen_filters=64, disc_filters=64):
        """
        Initialize CycleGAN.
        
        Args:
            img_shape: Shape of the images
            gen_filters: Number of filters in the first layer of the generator
            disc_filters: Number of filters in the first layer of the discriminator
        """
        self.img_shape = img_shape
        self.gen_filters = gen_filters
        self.disc_filters = disc_filters
        
        # Calculate output shape for the discriminator
        patch = int(img_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)
        
        # Number of filters in the first layer of generator and discriminator
        self.gf = gen_filters
        self.df = disc_filters
        
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        
        # Build the generators
        self.g_AB = self.build_generator()  # A (normal) -> B (alzheimer)
        self.g_BA = self.build_generator()  # B (alzheimer) -> A (normal)
        
        # Set up the combined model
        self.setup_combined_model()
    
    def build_generator(self):
        """
        Build generator network using the U-Net architecture.
        
        Returns:
            Generator model
        """
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = Sequential()
            result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))
            
            if apply_batchnorm:
                result.add(layers.BatchNormalization())
            
            result.add(layers.LeakyReLU())
            
            return result
        
        def upsample(filters, size, apply_dropout=False):
            initializer = tf.random_normal_initializer(0., 0.02)
            
            result = Sequential()
            result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False))
            
            result.add(layers.BatchNormalization())
            
            if apply_dropout:
                result.add(layers.Dropout(0.5))
            
            result.add(layers.ReLU())
            
            return result
        
        # Input layer
        inputs = layers.Input(shape=self.img_shape)
        
        # Downsampling
        down_stack = [
            downsample(self.gf, 4, apply_batchnorm=False),      # (bs, 64, 64, 64)
            downsample(self.gf * 2, 4),                         # (bs, 32, 32, 128)
            downsample(self.gf * 4, 4),                         # (bs, 16, 16, 256)
            downsample(self.gf * 8, 4),                         # (bs, 8, 8, 512)
            downsample(self.gf * 8, 4),                         # (bs, 4, 4, 512)
            downsample(self.gf * 8, 4),                         # (bs, 2, 2, 512)
            downsample(self.gf * 8, 4),                         # (bs, 1, 1, 512)
        ]
        
        # Upsampling
        up_stack = [
            upsample(self.gf * 8, 4, apply_dropout=True),       # (bs, 2, 2, 512)
            upsample(self.gf * 8, 4, apply_dropout=True),       # (bs, 4, 4, 512)
            upsample(self.gf * 8, 4, apply_dropout=True),       # (bs, 8, 8, 512)
            upsample(self.gf * 4, 4),                           # (bs, 16, 16, 256)
            upsample(self.gf * 2, 4),                           # (bs, 32, 32, 128)
            upsample(self.gf, 4),                               # (bs, 64, 64, 64)
        ]
        
        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(self.img_shape[2], 4, strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     activation='tanh')   # (bs, 128, 128, 3)
        
        x = inputs
        
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])
        
        x = last(x)
        
        return models.Model(inputs=inputs, outputs=x)
    
    def build_discriminator(self):
        """
        Build discriminator network.
        
        Returns:
            Discriminator model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        
        inputs = layers.Input(shape=self.img_shape)
        
        # First layer doesn't use batch normalization
        x = layers.Conv2D(self.df, 4, strides=2, padding='same',
                         kernel_initializer=initializer)(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        # Increase number of filters as we go deeper
        x = layers.Conv2D(self.df * 2, 4, strides=2, padding='same',
                         kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(self.df * 4, 4, strides=2, padding='same',
                         kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(self.df * 8, 4, strides=2, padding='same',
                         kernel_initializer=initializer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final layer to get to 1 dimension
        x = layers.Conv2D(1, 4, strides=1, padding='same',
                         kernel_initializer=initializer)(x)
        
        return models.Model(inputs=inputs, outputs=x)
    
    def setup_combined_model(self):
        """
        Set up the combined model that includes both generators and discriminators.
        """
        # Build discriminators
        self.d_A.compile(loss='mse',
                        optimizer=optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                        metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                        optimizer=optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                        metrics=['accuracy'])
        
        # For the combined model, we only want to train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        # Input images from both domains
        img_A = layers.Input(shape=self.img_shape)
        img_B = layers.Input(shape=self.img_shape)
        
        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        
        # Discriminators determine validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        # Combined model
        self.combined = models.Model(inputs=[img_A, img_B],
                                    outputs=[valid_A, valid_B,
                                            reconstr_A, reconstr_B,
                                            img_A_id, img_B_id])
        
        self.combined.compile(loss=['mse', 'mse',
                                   'mae', 'mae',
                                   'mae', 'mae'],
                             loss_weights=[1, 1, 10, 10, 1, 1],
                             optimizer=optimizers.Adam(learning_rate=2e-4, beta_1=0.5))
    
    def train(self, dataset_A, dataset_B, epochs, batch_size=1, sample_interval=50):
        """
        Train the CycleGAN.
        
        Args:
            dataset_A: Dataset of images from domain A (normal)
            dataset_B: Dataset of images from domain B (alzheimer)
            epochs: Number of epochs to train for
            batch_size: Batch size
            sample_interval: Interval between image samples
            
        Returns:
            Training history
        """
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        # Training history
        history = {
            'd_A_loss': [], 'd_B_loss': [], 
            'g_loss': [], 'adv_loss': [], 
            'reconstr_loss': [], 'id_loss': []
        }
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Select a random batch of images
            idx_A = np.random.randint(0, dataset_A.shape[0], batch_size)
            idx_B = np.random.randint(0, dataset_B.shape[0], batch_size)
            
            imgs_A = dataset_A[idx_A]
            imgs_B = dataset_B[idx_B]
            
            # ----------------------
            #  Train Discriminators
            # ----------------------
            
            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            
            # Train the discriminators
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
            
            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
            
            d_loss = 0.5 * np.add(dA_loss, dB_loss)
            
            # ------------------
            #  Train Generators
            # ------------------
            
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                 [valid, valid,
                                                 imgs_A, imgs_B,
                                                 imgs_A, imgs_B])
            
            elapsed_time = time.time() - start_time
            
            # Record history
            history['d_A_loss'].append(dA_loss[0])
            history['d_B_loss'].append(dB_loss[0])
            history['g_loss'].append(g_loss[0])
            history['adv_loss'].append(0.5 * np.add(g_loss[1], g_loss[2]))
            history['reconstr_loss'].append(0.5 * np.add(g_loss[3], g_loss[4]))
            history['id_loss'].append(0.5 * np.add(g_loss[5], g_loss[6]))
            
            # Print progress
            print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f}, adv: {g_loss[1]+g_loss[2]:.4f}, recon: {g_loss[3]+g_loss[4]:.4f}, id: {g_loss[5]+g_loss[6]:.4f}] time: {elapsed_time:.2f}s")
            
            # If at save interval, save generated samples
            if (epoch + 1) % sample_interval == 0:
                self.sample_images(epoch+1, imgs_A, imgs_B)
        
        return history
    
    def sample_images(self, epoch, imgs_A=None, imgs_B=None, save_dir="generated_samples"):
        """
        Save sample images from the generator during training.
        
        Args:
            epoch: Current epoch
            imgs_A: Sample images from domain A (normal)
            imgs_B: Sample images from domain B (alzheimer)
            save_dir: Directory to save generated images
        """
        os.makedirs(save_dir, exist_ok=True)
        r, c = 2, 3
        
        if imgs_A is None or imgs_B is None:
            return
        
        # Demo on only 1 image from each domain
        if len(imgs_A.shape) == 4 and imgs_A.shape[0] > 1:
            imgs_A = imgs_A[:1]
        if len(imgs_B.shape) == 4 and imgs_B.shape[0] > 1:
            imgs_B = imgs_B[:1]
        
        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        
        # Generate images
        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        # Create figure
        fig, axs = plt.subplots(r, c, figsize=(12, 6))
        
        titles = ['Original', 'Translated', 'Reconstructed']
        
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[i*c + j])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
        
        fig.suptitle(f"CycleGAN - Epoch {epoch}")
        fig.savefig(f"{save_dir}/cyclegan_epoch_{epoch}.png")
        plt.close()
    
    def generate_alzheimer_images(self, normal_images):
        """
        Generate Alzheimer's versions of normal brain images.
        
        Args:
            normal_images: Normal brain images
            
        Returns:
            Generated Alzheimer's brain images
        """
        return self.g_AB.predict(normal_images)
    
    def generate_normal_images(self, alzheimer_images):
        """
        Generate normal versions of Alzheimer's brain images.
        
        Args:
            alzheimer_images: Alzheimer's brain images
            
        Returns:
            Generated normal brain images
        """
        return self.g_BA.predict(alzheimer_images)
    
    def save_models(self, save_dir="saved_models"):
        """
        Save the generator and discriminator models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        self.g_AB.save(f"{save_dir}/cyclegan_g_AB.h5")
        self.g_BA.save(f"{save_dir}/cyclegan_g_BA.h5")
        self.d_A.save(f"{save_dir}/cyclegan_d_A.h5")
        self.d_B.save(f"{save_dir}/cyclegan_d_B.h5")
    
    def load_models(self, g_AB_path, g_BA_path, d_A_path, d_B_path):
        """
        Load saved generator and discriminator models.
        
        Args:
            g_AB_path: Path to saved generator A->B model
            g_BA_path: Path to saved generator B->A model
            d_A_path: Path to saved discriminator A model
            d_B_path: Path to saved discriminator B model
        """
        self.g_AB = models.load_model(g_AB_path)
        self.g_BA = models.load_model(g_BA_path)
        self.d_A = models.load_model(d_A_path)
        self.d_B = models.load_model(d_B_path)
        
        # Setup the combined model again
        self.setup_combined_model() 