import os
import numpy as np
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_and_preprocess_images(data_dir, img_size=(128, 128), batch_size=32):
    """
    Load and preprocess images from a directory.
    
    Args:
        data_dir: Directory containing class subdirectories of images
        img_size: Target size for the images
        batch_size: Batch size for the dataset
        
    Returns:
        tf.data.Dataset: Preprocessed dataset
    """
    # Create dataset from directory
    dataset = keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    # Normalize images to [-1, 1] for GAN training
    normalization_layer = keras.layers.Rescaling(scale=1./127.5, offset=-1)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    return dataset

def create_paired_dataset(normal_dir, alzheimer_dir, img_size=(128, 128), test_split=0.2):
    """
    Create paired datasets for CycleGAN.
    
    Args:
        normal_dir: Directory containing normal brain images
        alzheimer_dir: Directory containing Alzheimer's brain images
        img_size: Target size for the images
        test_split: Fraction of data to use for testing
        
    Returns:
        Tuple of train and test datasets for both domains
    """
    normal_images = []
    alzheimer_images = []
    
    # Load normal images
    for img_file in tqdm(os.listdir(normal_dir)):
        img_path = os.path.join(normal_dir, img_file)
        img = img_to_array(load_img(img_path, target_size=img_size))
        normal_images.append(img)
        
    # Load Alzheimer's images
    for img_file in tqdm(os.listdir(alzheimer_dir)):
        img_path = os.path.join(alzheimer_dir, img_file)
        img = img_to_array(load_img(img_path, target_size=img_size))
        alzheimer_images.append(img)
    
    # Convert to numpy arrays and normalize to [-1, 1]
    normal_images = np.array(normal_images) / 127.5 - 1
    alzheimer_images = np.array(alzheimer_images) / 127.5 - 1
    
    # Split into train and test sets
    normal_train, normal_test = train_test_split(normal_images, test_size=test_split, random_state=42)
    alzheimer_train, alzheimer_test = train_test_split(alzheimer_images, test_size=test_split, random_state=42)
    
    return (normal_train, normal_test, alzheimer_train, alzheimer_test)

def generate_low_res_images(images, scale_factor=4):
    """
    Generate low-resolution images for super-resolution training.
    
    Args:
        images: High-resolution images (numpy array)
        scale_factor: Downsampling factor
        
    Returns:
        Numpy array of low-resolution images
    """
    low_res_images = []
    
    for img in images:
        h, w, _ = img.shape
        low_res_h, low_res_w = h // scale_factor, w // scale_factor
        low_res_img = cv2.resize(img, (low_res_w, low_res_h), interpolation=cv2.INTER_CUBIC)
        low_res_img = cv2.resize(low_res_img, (w, h), interpolation=cv2.INTER_CUBIC)
        low_res_images.append(low_res_img)
    
    return np.array(low_res_images)

def visualize_results(original, generated, titles=None, save_path=None):
    """
    Visualize original and generated images side by side.
    
    Args:
        original: Original image or batch of images
        generated: Generated image or batch of images
        titles: List of titles for the images
        save_path: Path to save the visualization
    """
    if len(original.shape) == 4:
        # Display first image from batch
        original = original[0]
        generated = generated[0]
    
    # Rescale images from [-1, 1] to [0, 1]
    original = (original + 1) / 2.0
    generated = (generated + 1) / 2.0
    
    plt.figure(figsize=(12, 6))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.axis('off')
    if titles:
        plt.title(titles[0])
    else:
        plt.title('Original')
    
    # Plot generated image
    plt.subplot(1, 2, 2)
    plt.imshow(generated)
    plt.axis('off')
    if titles and len(titles) > 1:
        plt.title(titles[1])
    else:
        plt.title('Generated')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show() 