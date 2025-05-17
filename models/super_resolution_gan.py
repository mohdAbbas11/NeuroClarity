import tensorflow as tf
from keras import layers, models, optimizers, applications
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class SRGAN:
    def __init__(self, img_shape=(128, 128, 3), scale_factor=4):
        """
        Initialize Super-Resolution GAN.
        
        Args:
            img_shape: Shape of the high-resolution images
            scale_factor: Scale factor for super-resolution
        """
        self.img_shape = img_shape
        self.scale_factor = scale_factor
        
        # Calculate low-resolution shape
        self.lr_shape = (img_shape[0] // scale_factor, img_shape[1] // scale_factor, img_shape[2])
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # Input low-resolution image
        lr_input = layers.Input(shape=self.lr_shape)
        
        # Generate high-resolution image
        gen_hr = self.generator(lr_input)
        
        # Discriminate generated high-resolution image
        valid = self.discriminator(gen_hr)
        
        # VGG feature extractor for perceptual loss
        self.vgg = self.build_vgg_feature_extractor()
        self.vgg.trainable = False
        
        # Get VGG features for fake and real HR images
        fake_features = self.vgg(gen_hr)
        
        # Combined model
        self.combined = models.Model(lr_input, [valid, fake_features])
        self.combined.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1],
            optimizer=optimizers.Adam(learning_rate=1e-4)
        )
    
    def build_generator(self):
        """
        Build generator network with residual blocks.
        
        Returns:
            Generator model
        """
        def residual_block(x, filters):
            """Residual block for the generator"""
            shortcut = x
            
            x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.PReLU(shared_axes=[1, 2])(x)
            
            x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Add()([shortcut, x])
            
            return x
        
        def upsample_block(x, filters):
            """Upsampling block using Conv2D and PixelShuffle"""
            x = layers.Conv2D(filters * 4, kernel_size=3, strides=1, padding='same')(x)
            x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)  # PixelShuffle
            x = layers.PReLU(shared_axes=[1, 2])(x)
            
            return x
        
        # Input low-resolution image
        lr_input = layers.Input(shape=self.lr_shape)
        
        # Initial convolutional layer
        x = layers.Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        
        # Save the output of the first conv layer
        conv1_out = x
        
        # Add residual blocks
        num_residual_blocks = 16
        for _ in range(num_residual_blocks):
            x = residual_block(x, 64)
        
        # Add convolutional layer after residual blocks
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add the output of the first conv layer
        x = layers.Add()([conv1_out, x])
        
        # Upsampling layers (number depends on scale_factor)
        num_upsample_blocks = self.scale_factor // 2
        for _ in range(num_upsample_blocks):
            x = upsample_block(x, 64)
        
        # Final output layer
        x = layers.Conv2D(self.img_shape[2], kernel_size=9, strides=1, padding='same', activation='tanh')(x)
        
        return models.Model(lr_input, x)
    
    def build_discriminator(self):
        """
        Build discriminator network.
        
        Returns:
            Discriminator model
        """
        def discriminator_block(x, filters, strides=1, bn=True):
            """Discriminator block with Conv, BatchNorm and LeakyReLU"""
            x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
            if bn:
                x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            
            return x
        
        # Input high-resolution image
        img_input = layers.Input(shape=self.img_shape)
        
        # Series of discriminator blocks
        x = discriminator_block(img_input, 64, bn=False)
        x = discriminator_block(x, 64, strides=2)
        x = discriminator_block(x, 128)
        x = discriminator_block(x, 128, strides=2)
        x = discriminator_block(x, 256)
        x = discriminator_block(x, 256, strides=2)
        x = discriminator_block(x, 512)
        x = discriminator_block(x, 512, strides=2)
        
        # Dense layers for classification
        x = layers.Flatten()(x)
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Output layer with sigmoid activation
        validity = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(img_input, validity)
    
    def build_vgg_feature_extractor(self):
        """
        Build VGG19 feature extractor for content loss.
        
        Returns:
            VGG feature extractor model
        """
        # Use pre-trained VGG19 model from Keras applications
        vgg = applications.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_shape
        )
        
        # Set all layers to non-trainable
        vgg.trainable = False
        
        # Create a model that returns the output of the 5th convolutional layer from the 4th block
        feature_model = models.Model(
            inputs=vgg.input,
            outputs=vgg.get_layer('block5_conv4').output
        )
        
        # Create a new input for normalized values ([-1, 1] -> [0, 255])
        img_input = layers.Input(shape=self.img_shape)
        
        # Normalize the input to VGG19 expected range
        preprocessed_input = layers.Lambda(
            lambda x: (x + 1) * 127.5
        )(img_input)
        
        # Get features
        features = feature_model(preprocessed_input)
        
        return models.Model(img_input, features)
    
    def train(self, hr_images, epochs, batch_size=1, sample_interval=50):
        """
        Train the SRGAN.
        
        Args:
            hr_images: High-resolution training images
            epochs: Number of epochs to train for
            batch_size: Batch size
            sample_interval: Interval between image samples
            
        Returns:
            Training history
        """
        # Create low-resolution versions of the images
        lr_images = np.array([cv2.resize(
            img,
            (self.lr_shape[1], self.lr_shape[0]),
            interpolation=cv2.INTER_CUBIC
        ) for img in hr_images])
        
        # Normalize images to [-1, 1]
        lr_images = lr_images / 127.5 - 1
        hr_images = hr_images / 127.5 - 1
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training history
        history = {
            'd_loss': [],
            'd_acc': [],
            'g_loss': [],
            'g_adv_loss': [],
            'g_content_loss': []
        }
        
        for epoch in range(epochs):
            # ----------------------
            #  Train Discriminator
            # ----------------------
            
            # Sample a batch of images
            idx = np.random.randint(0, hr_images.shape[0], batch_size)
            hr_imgs = hr_images[idx]
            lr_imgs = lr_images[idx]
            
            # Generate high-resolution images from low-resolution ones
            gen_hr = self.generator.predict(lr_imgs)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(hr_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ------------------
            #  Train Generator
            # ------------------
            
            # Extract VGG features for real HR images
            hr_features = self.vgg.predict(hr_imgs)
            
            # Train the generator
            g_loss = self.combined.train_on_batch(lr_imgs, [valid, hr_features])
            
            # Record history
            history['d_loss'].append(d_loss[0])
            history['d_acc'].append(d_loss[1])
            history['g_loss'].append(g_loss[0])
            history['g_adv_loss'].append(g_loss[1])
            history['g_content_loss'].append(g_loss[2])
            
            # Print progress
            print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f}, adv: {g_loss[1]:.4f}, content: {g_loss[2]:.4f}]")
            
            # If at save interval, save generated image samples
            if (epoch + 1) % sample_interval == 0:
                self.sample_images(epoch + 1, hr_imgs, lr_imgs)
        
        return history
    
    def sample_images(self, epoch, hr_imgs, lr_imgs, save_dir="generated_samples"):
        """
        Save sample images from the generator during training.
        
        Args:
            epoch: Current epoch
            hr_imgs: High-resolution images
            lr_imgs: Low-resolution images
            save_dir: Directory to save generated images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Demo on just the first image from the batch
        if len(hr_imgs.shape) == 4 and hr_imgs.shape[0] > 1:
            hr_imgs = hr_imgs[:1]
            lr_imgs = lr_imgs[:1]
        
        # Generate SR images
        gen_hr = self.generator.predict(lr_imgs)
        
        # Create bicubic upsampled version for comparison
        bicubic_hr = np.array([cv2.resize(
            lr_img,
            (self.img_shape[1], self.img_shape[0]),
            interpolation=cv2.INTER_CUBIC
        ) for lr_img in lr_imgs])
        
        # Rescale images 0 - 1
        gen_hr = 0.5 * gen_hr + 0.5
        hr_imgs = 0.5 * hr_imgs + 0.5
        lr_imgs = 0.5 * lr_imgs + 0.5
        bicubic_hr = 0.5 * bicubic_hr + 0.5
        
        # Plot all images
        plt.figure(figsize=(16, 4))
        
        plt.subplot(141)
        plt.imshow(lr_imgs[0])
        plt.title('Low Resolution')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(bicubic_hr[0])
        plt.title('Bicubic Upsampled')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(gen_hr[0])
        plt.title('SRGAN Generated')
        plt.axis('off')
        
        plt.subplot(144)
        plt.imshow(hr_imgs[0])
        plt.title('Original HR')
        plt.axis('off')
        
        plt.suptitle(f"Super-Resolution GAN - Epoch {epoch}")
        plt.savefig(f"{save_dir}/srgan_epoch_{epoch}.png")
        plt.close()
    
    def generate_high_res(self, lr_images):
        """
        Generate high-resolution images from low-resolution ones.
        
        Args:
            lr_images: Low-resolution images (normalized to [-1, 1])
            
        Returns:
            High-resolution images
        """
        return self.generator.predict(lr_images)
    
    def save_models(self, save_dir="saved_models"):
        """
        Save the generator and discriminator models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        self.generator.save(f"{save_dir}/srgan_generator.h5")
        self.discriminator.save(f"{save_dir}/srgan_discriminator.h5")
    
    def load_models(self, generator_path, discriminator_path):
        """
        Load saved generator and discriminator models.
        
        Args:
            generator_path: Path to saved generator model
            discriminator_path: Path to saved discriminator model
        """
        self.generator = models.load_model(generator_path)
        self.discriminator = models.load_model(discriminator_path)
        
        # Rebuild the combined model
        lr_input = layers.Input(shape=self.lr_shape)
        gen_hr = self.generator(lr_input)
        
        self.discriminator.trainable = False
        valid = self.discriminator(gen_hr)
        fake_features = self.vgg(gen_hr)
        
        self.combined = models.Model(lr_input, [valid, fake_features])
        self.combined.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1],
            optimizer=optimizers.Adam(learning_rate=1e-4)
        ) 