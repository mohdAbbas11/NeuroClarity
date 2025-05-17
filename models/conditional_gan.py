import tensorflow as tf
from keras import layers, models, optimizers, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
from utils.data_processing import load_and_preprocess_images

class ConditionalGAN:
    def __init__(self, img_shape=(128, 128, 3), latent_dim=100, num_classes=2):
        """
        Initialize the Conditional GAN.
        
        Args:
            img_shape: Shape of the input images
            latent_dim: Dimension of the latent space
            num_classes: Number of classes for conditioning
        """
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.num_classes,))
        img = self.generator([noise, label])
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])
        
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model([noise, label], valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.5)  # Lower learning rate for generator
        )
    
    def build_generator(self):
        """
        Build the generator network.
        
        Returns:
            Generator model
        """
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.num_classes,))
        
        # Concatenate noise and label
        model_input = layers.Concatenate()([noise, label])
        
        # Initial dense layer with increased capacity
        x = layers.Dense(8 * 8 * 1024, use_bias=False)(model_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Reshape((8, 8, 1024))(x)
        
        # Upsampling blocks with residual connections
        # Block 1
        x1 = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.LeakyReLU(0.2)(x1)
        
        # Block 2
        x2 = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.LeakyReLU(0.2)(x2)
        
        # Block 3
        x3 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.LeakyReLU(0.2)(x3)
        
        # Block 4
        x4 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.LeakyReLU(0.2)(x4)
        
        # Output layer with tanh activation
        img = layers.Conv2D(self.img_shape[2], kernel_size=3, padding='same', activation='tanh')(x4)
        
        return models.Model([noise, label], img)
    
    def build_discriminator(self):
        """
        Build the discriminator network.
        
        Returns:
            Discriminator model
        """
        img = layers.Input(shape=self.img_shape)
        label = layers.Input(shape=(self.num_classes,))
        
        # Process the image
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(img)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Flatten()(x)
        
        # Process the label
        label_embedding = layers.Dense(512)(label)
        label_embedding = layers.LeakyReLU(0.2)(label_embedding)
        
        # Concatenate image features and label embedding
        combined = layers.Concatenate()([x, label_embedding])
        
        # Final layers
        x = layers.Dense(1024)(combined)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer with sigmoid activation
        validity = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model([img, label], validity)
    
    def train(self, X_train, y_train, epochs, batch_size=32, sample_interval=200):
        """
        Train the Conditional GAN.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            epochs: Number of epochs to train for
            batch_size: Batch size
            sample_interval: Interval between image samples
            
        Returns:
            Training history
        """
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        d_loss_history = []
        g_loss_history = []
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = y_train[idx]
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator([noise, labels], training=True)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate a batch of new labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size)
            sampled_labels = utils.to_categorical(sampled_labels, self.num_classes)
            
            # Train the generator (twice to help with stability)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            
            # Record history
            d_loss_history.append(d_loss[0])
            g_loss_history.append(g_loss)
            
            # Print the progress
            print(f"Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            
            # If at save interval => save generated image samples
            if (epoch + 1) % sample_interval == 0:
                self.sample_images(epoch + 1)
        
        return {"d_loss": d_loss_history, "g_loss": g_loss_history}
    
    def sample_images(self, epoch, save_dir="generated_samples"):
        """
        Generate and save sample images during training.
        
        Args:
            epoch: Current epoch number
            save_dir: Directory to save the generated images
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate images for each class
        for class_idx in range(self.num_classes):
            # Generate a batch of noise
            noise = np.random.normal(0, 1, (5, self.latent_dim))
            
            # Generate labels for the current class
            labels = np.zeros((5, self.num_classes))
            labels[:, class_idx] = 1
            
            # Generate images
            gen_imgs = self.generator([noise, labels], training=False)
            
            # Rescale images from [-1, 1] to [0, 1]
            gen_imgs = 0.5 * gen_imgs + 0.5
            
            # Save the generated images
            for i in range(5):
                plt.figure(figsize=(2, 2))
                plt.imshow(gen_imgs[i])
                plt.axis('off')
                plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_class_{class_idx}_sample_{i}.png"))
                plt.close()
    
    def generate_samples(self, n_samples=10, labels=None):
        """
        Generate samples from the trained generator.
        
        Args:
            n_samples: Number of samples to generate
            labels: Optional specific labels to generate for
            
        Returns:
            Generated images
        """
        if labels is None:
            # Generate random labels
            labels = np.random.randint(0, self.num_classes, n_samples)
            labels = utils.to_categorical(labels, self.num_classes)
        
        # Generate noise
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Generate images
        gen_imgs = self.generator([noise, labels], training=False)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        return gen_imgs
    
    def save_models(self, save_dir="saved_models"):
        """
        Save the generator and discriminator models.
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        self.generator.save(f"{save_dir}/cgan_generator.h5")
        self.discriminator.save(f"{save_dir}/cgan_discriminator.h5")
    
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
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.num_classes,))
        img = self.generator([noise, label])
        
        self.discriminator.trainable = False
        valid = self.discriminator([img, label])
        
        self.combined = models.Model([noise, label], valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Test accuracy
        """
        return self.combined.evaluate(X_test, y_test)[1]
    
    def evaluate_model(self, X_test, y_test, n_samples=10):
        """
        Evaluate the Conditional GAN model performance.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            n_samples: Number of samples to generate for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # 1. Evaluate discriminator on real images
        real_scores = self.discriminator.predict([X_test, y_test])
        real_accuracy = np.mean(real_scores > 0.5)
        
        # 2. Generate fake images and evaluate discriminator
        noise = np.random.normal(0, 1, (len(X_test), self.latent_dim))
        fake_images = self.generator.predict([noise, y_test])
        fake_scores = self.discriminator.predict([fake_images, y_test])
        fake_accuracy = np.mean(fake_scores < 0.5)
        
        # 3. Calculate FID score (Fréchet Inception Distance)
        try:
            from keras.applications.inception_v3 import InceptionV3, preprocess_input
            from scipy import linalg
            
            # Load Inception model
            inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
            
            # Preprocess images for Inception
            def preprocess_for_inception(images):
                images = tf.image.resize(images, (299, 299))
                images = preprocess_input(images)
                return images
            
            # Get features for real and fake images
            real_features = inception.predict(preprocess_for_inception(X_test))
            fake_features = inception.predict(preprocess_for_inception(fake_images))
            
            # Calculate mean and covariance
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # Calculate FID
            ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
            covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
        except:
            fid = None
        
        # 4. Generate sample images for visual evaluation
        sample_images = self.generate_samples(n_samples=n_samples)
        
        # 5. Calculate diversity score (average pairwise L2 distance between generated images)
        diversity = np.mean([np.mean(np.square(sample_images[i] - sample_images[j])) 
                           for i in range(n_samples) 
                           for j in range(i+1, n_samples)])
        
        return {
            'discriminator_real_accuracy': float(real_accuracy),
            'discriminator_fake_accuracy': float(fake_accuracy),
            'fid_score': float(fid) if fid is not None else None,
            'diversity_score': float(diversity),
            'sample_images': sample_images
        }
    
    def plot_evaluation_results(self, eval_results, save_path=None):
        """
        Plot evaluation results.
        
        Args:
            eval_results: Dictionary containing evaluation metrics
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Plot discriminator accuracies
        plt.subplot(1, 3, 1)
        accuracies = [eval_results['discriminator_real_accuracy'], 
                     eval_results['discriminator_fake_accuracy']]
        plt.bar(['Real', 'Fake'], accuracies)
        plt.title('Discriminator Accuracy')
        plt.ylim(0, 1)
        
        # Plot sample images
        plt.subplot(1, 3, 2)
        sample_images = eval_results['sample_images']
        n_samples = len(sample_images)
        for i in range(min(4, n_samples)):
            plt.subplot(2, 2, i+1)
            plt.imshow((sample_images[i] + 1) / 2)  # Convert from [-1,1] to [0,1]
            plt.axis('off')
        plt.suptitle('Generated Samples')
        
        # Plot metrics
        plt.subplot(1, 3, 3)
        metrics = {
            'FID Score': eval_results['fid_score'],
            'Diversity': eval_results['diversity_score']
        }
        plt.bar(metrics.keys(), [m for m in metrics.values() if m is not None])
        plt.title('Quality Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Evaluation results plot saved to {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create an instance of ConditionalGAN
    cgan = ConditionalGAN()

    # Load and preprocess training data
    X_train, y_train = load_and_preprocess_images('data_cgan/train', img_size=128, label_mode='categorical')
    
    # Load test data
    X_test, y_test = load_and_preprocess_images('data_cgan/test', img_size=128, label_mode='categorical')

    # Train the model
    history = cgan.train(X_train, y_train, epochs=100, batch_size=32, sample_interval=200)

    # Evaluate the model
    eval_results = cgan.evaluate_model(X_test, y_test, n_samples=10)
    
    # Plot evaluation results
    cgan.plot_evaluation_results(eval_results, save_path='evaluation_results.png')
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"Discriminator Real Accuracy: {eval_results['discriminator_real_accuracy']:.4f}")
    print(f"Discriminator Fake Accuracy: {eval_results['discriminator_fake_accuracy']:.4f}")
    if eval_results['fid_score'] is not None:
        print(f"FID Score: {eval_results['fid_score']:.4f}")
    print(f"Diversity Score: {eval_results['diversity_score']:.4f}") 