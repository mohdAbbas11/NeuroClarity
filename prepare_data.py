#!/usr/bin/env python3
"""
Script to prepare sample data for testing the Alzheimer's detection application.
This script generates placeholder data without requiring Kaggle credentials.
"""

import os
import numpy as np
from PIL import Image
import argparse
import shutil
from tqdm import tqdm

def create_directory_structure(base_dir="data"):
    """Create the necessary directory structure for the application."""
    
    # Create main directories
    directories = {
        'normal': os.path.join(base_dir, 'normal'),
        'alzheimer': os.path.join(base_dir, 'alzheimer'),
        'high_res': os.path.join(base_dir, 'high_res'),
        'models': os.path.join(base_dir, 'models'),
        # Add multi-class directories
        'NonDemented': os.path.join(base_dir, 'NonDemented'),
        'VeryMildDemented': os.path.join(base_dir, 'VeryMildDemented'),
        'MildDemented': os.path.join(base_dir, 'MildDemented'),
        'ModerateDemented': os.path.join(base_dir, 'ModerateDemented')
    }
    
    # Create each directory
    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create results directory for the classifier
    os.makedirs(os.path.join(base_dir, '..', 'results'), exist_ok=True)
    print(f"Created directory: {os.path.join(base_dir, '..', 'results')}")
    
    return directories

def generate_sample_brain_images(directories, num_samples=10, size=(128, 128)):
    """Generate sample brain-like images for testing."""
    
    print(f"Generating {num_samples} sample images for each category...")
    
    # Generate normal brain images (darker with less noise)
    for i in tqdm(range(num_samples), desc="Generating normal brain images"):
        # Create a circular mask for brain-like shape
        img = np.zeros(size + (3,), dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 2 - 5
        
        # Create a circular brain-like shape
        for x in range(size[0]):
            for y in range(size[1]):
                if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                    # Normal brains: darker gray values
                    img[x, y] = [
                        np.random.randint(70, 110),  # Gray matter
                        np.random.randint(70, 110),
                        np.random.randint(70, 110)
                    ]
        
        # Add some structures to make it look more brain-like
        for _ in range(5):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(5, 15)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(120, 150)] * 3  # Brighter structures
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(directories['normal'], f"normal_sample_{i}.jpg"))
        pil_img.save(os.path.join(directories['NonDemented'], f"nondemented_sample_{i}.jpg"))
        
        # Also save to high_res for SRGAN
        pil_img.save(os.path.join(directories['high_res'], f"normal_sample_{i}.jpg"))
    
    # Generate Very Mild Demented images (slightly altered)
    for i in tqdm(range(num_samples), desc="Generating very mild demented images"):
        # Create a circular mask for brain-like shape
        img = np.zeros(size + (3,), dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 2 - 5
        
        # Create a circular brain-like shape
        for x in range(size[0]):
            for y in range(size[1]):
                if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                    # Very mild: slight variation in gray values
                    img[x, y] = [
                        np.random.randint(65, 115),  # Slightly more variable gray matter
                        np.random.randint(65, 115),
                        np.random.randint(65, 115)
                    ]
        
        # Add some structures to make it look more brain-like
        for _ in range(5):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(5, 15)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(110, 160)] * 3  # Brighter structures
        
        # Add 1-2 small bright spots (very mild changes)
        for _ in range(np.random.randint(1, 3)):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(2, 4)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(180, 220)] * 3  # Mild bright spots
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(directories['VeryMildDemented'], f"verymild_sample_{i}.jpg"))
    
    # Generate Mild Demented images (more noticeable changes)
    for i in tqdm(range(num_samples), desc="Generating mild demented images"):
        # Create a circular mask for brain-like shape
        img = np.zeros(size + (3,), dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 2 - 5
        
        # Create a circular brain-like shape
        for x in range(size[0]):
            for y in range(size[1]):
                if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                    # Mild demented: more variable gray values
                    img[x, y] = [
                        np.random.randint(60, 120),  # More variable gray matter
                        np.random.randint(60, 120),
                        np.random.randint(60, 120)
                    ]
        
        # Add some structures to make it look more brain-like
        for _ in range(5):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(5, 15)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(100, 180)] * 3  # Brighter structures
        
        # Add 3-5 medium bright spots (mild changes)
        for _ in range(np.random.randint(3, 6)):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(3, 5)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(190, 230)] * 3  # Medium bright spots
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(directories['MildDemented'], f"mild_sample_{i}.jpg"))
    
    # Generate Moderate/Alzheimer's brain images (more noise, significant bright spots)
    for i in tqdm(range(num_samples), desc="Generating moderate demented images"):
        # Create a circular mask for brain-like shape
        img = np.zeros(size + (3,), dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 2 - 5
        
        # Create a circular brain-like shape
        for x in range(size[0]):
            for y in range(size[1]):
                if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                    # Alzheimer's brains: more varied gray values
                    img[x, y] = [
                        np.random.randint(60, 120),  # More variable gray matter
                        np.random.randint(60, 120),
                        np.random.randint(60, 120)
                    ]
        
        # Add some structures to make it look more brain-like
        for _ in range(5):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(5, 15)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(100, 180)] * 3  # Brighter structures
        
        # Add abnormal bright spots (representing amyloid plaques or other pathology)
        for _ in range(np.random.randint(6, 10)):
            x, y = np.random.randint(center[0]-radius//2, center[0]+radius//2), np.random.randint(center[1]-radius//2, center[1]+radius//2)
            r = np.random.randint(3, 8)
            for dx in range(-r, r):
                for dy in range(-r, r):
                    if dx**2 + dy**2 < r**2 and 0 <= x+dx < size[0] and 0 <= y+dy < size[1]:
                        img[x+dx, y+dy] = [np.random.randint(200, 255)] * 3  # Bright spots
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(directories['alzheimer'], f"alzheimer_sample_{i}.jpg"))
        pil_img.save(os.path.join(directories['ModerateDemented'], f"moderate_sample_{i}.jpg"))
    
    print(f"Created {num_samples} sample images in each category.")

def create_placeholder_models(directories):
    """Create empty model files to allow the app to run without actual trained models."""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create empty placeholder files
    model_files = [
        "conditional_gan_model.h5",
        "cyclegan_generator_A2B.h5",
        "cyclegan_generator_B2A.h5",
        "srgan_generator.h5"
    ]
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            # Create an empty file
            with open(model_path, 'w') as f:
                f.write("# Placeholder model file\n")
            print(f"Created placeholder model file: {model_path}")
    
    # Create placeholder classifier model
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    classifier_model_path = os.path.join(results_dir, "alzheimer_classifier_model.h5")
    if not os.path.exists(classifier_model_path):
        with open(classifier_model_path, 'w') as f:
            f.write("# Placeholder classifier model file\n")
        print(f"Created placeholder classifier model file: {classifier_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare sample data for testing the Alzheimer's detection app")
    parser.add_argument('--output_dir', type=str, default="data", help="Directory to save the sample data")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of sample images to generate per category")
    parser.add_argument('--image_size', type=int, default=128, help="Size of the generated images (square)")
    
    args = parser.parse_args()
    
    print(f"Preparing sample data in directory: {args.output_dir}")
    
    # Create directory structure
    directories = create_directory_structure(args.output_dir)
    
    # Generate sample images
    generate_sample_brain_images(directories, num_samples=args.num_samples, size=(args.image_size, args.image_size))
    
    # Create placeholder model files
    create_placeholder_models(directories)
    
    print("\nSample data preparation complete!")
    print("You can now run the app with: streamlit run app.py")
    print("Note: The app will use placeholder data for demonstration purposes.")

if __name__ == "__main__":
    main() 