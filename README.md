# Alzheimer's Detection using GANs

This project uses various GAN architectures and deep learning classifiers for Alzheimer's disease detection and visualization through a Streamlit web application.

## GPU Setup Instructions

To ensure TensorFlow can utilize your GPU, follow these setup instructions:

### Prerequisites

1. **NVIDIA GPU** with CUDA capability
2. **NVIDIA Drivers** (latest version recommended)
3. **CUDA Toolkit** (version 11.2 compatible with TensorFlow 2.12)
4. **cuDNN** (version 8.1)
5. **Python 3.8-3.10** (TensorFlow 2.12 is not compatible with Python 3.11+)

### Setting Environment Variables

#### Option 1: Windows Batch File (Recommended)

1. Run the provided batch file before starting the application:
   ```
   .\run_app.bat
   ```

This script automatically:
- Sets up the necessary environment variables
- Checks for GPU detection
- Installs required dependencies
- Runs the Streamlit application

### Verifying GPU Detection

To verify that TensorFlow can detect and use your GPU:
```
python verify_gpu.py
```

### Troubleshooting GPU Issues

If TensorFlow cannot detect your GPU:

1. **Check CUDA installation**:
   - Ensure CUDA 11.2 is installed
   - Verify cuDNN 8.1 is properly set up
   - Check that PATH variables point to correct CUDA directories

2. **Check TensorFlow installation**:
   - Ensure you have the GPU version: `pip install tensorflow`
   - Check compatibility between TensorFlow, CUDA, and cuDNN versions

3. **Driver issues**:
   - Update NVIDIA drivers to the latest version
   - Run `nvidia-smi` to verify drivers are working

4. **Environment variables**:
   - Set these variables before running TensorFlow:
     ```
     CUDA_VISIBLE_DEVICES=0
     TF_FORCE_GPU_ALLOW_GROWTH=true
     TF_GPU_ALLOCATOR=cuda_malloc_async
     ```

## About the Application

This application demonstrates multiple approaches to Alzheimer's disease detection and visualization:

1. **GAN Models**:
   - **Conditional GAN**: Generate brain images conditioned on being normal or having Alzheimer's disease
   - **CycleGAN**: Transform normal brain images to Alzheimer's disease and vice versa
   - **Super-Resolution GAN**: Enhance low-resolution brain images for better diagnosis

2. **New! Multi-class Alzheimer's Classifier**:
   - Detects different stages of Alzheimer's disease:
     - Non-Demented (normal)
     - Very Mild Demented
     - Mild Demented
     - Moderate Demented
   - Built with transfer learning using EfficientNetB0
   - Provides confidence scores and visualizations

## Training the Classifier

To train the multi-class Alzheimer's classifier on your data:

```
python train_classifier.py --data_dir data --epochs 20 --fine_tune_epochs 10 --batch_size 32
```

Arguments:
- `--data_dir`: Base directory containing class subdirectories
- `--epochs`: Number of epochs for initial training
- `--fine_tune_epochs`: Number of epochs for fine-tuning
- `--batch_size`: Batch size for training
- `--img_size`: Image size (default: 128)
- `--no_augment`: Disable data augmentation
- `--output_dir`: Directory to save results (default: results/)

## Making Predictions

To classify brain scans using the trained model:

```
python predict.py --model_path results/alzheimer_classifier_model.h5 --image_path path/to/image.jpg
```

For batch processing:
```
python predict.py --model_path results/alzheimer_classifier_model.h5 --image_path path/to/directory --batch
```

## Running the Application

Simply run:
```
streamlit run app.py
```

Or use the all-in-one script:
```
.\run_app.bat
```

For any issues or questions, please contact the project maintainers.

## Quick Start

To run the complete workflow (download data, train all models, and launch the interface):
```
python main.py --kaggle_username YOUR_USERNAME --kaggle_key YOUR_KEY
```

You can skip specific steps if needed:
```
python main.py --skip_download --skip_training  # Only run the interface
python main.py --skip_training  # Download data and run interface
python main.py --epochs 10  # Run with fewer training epochs
```

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Download Alzheimer's dataset from Kaggle:
```
python download_dataset.py
```

This script will:
- Download a standard Alzheimer's dataset with 4 classes from Kaggle
- Organize it into the required directory structure
- Prepare data for all three GAN models

Note: You need a Kaggle account and API key to download datasets automatically. You can:
- Provide credentials directly: `python download_dataset.py --kaggle_username YOUR_USERNAME --kaggle_key YOUR_KEY`
- Or set up credentials manually by downloading the kaggle.json file from your account and placing it in ~/.kaggle/

Alternatively, you can manually create the directory structure:
```
mkdir -p data/normal data/alzheimer data/high_res
```

3. Place your brain scan images in the appropriate directories:
   - `data/normal/`: Normal brain images
   - `data/alzheimer/`: Alzheimer's brain images
   - `data/high_res/`: High-resolution brain images for SRGAN training
   
   Note: For the Conditional GAN, organize your data with class subdirectories:
   ```
   data/
   ├── normal/
   └── alzheimer/
   ```

## Training Models

### Train Conditional GAN
```
# First, create a directory with only normal and alzheimer subfolders
mkdir data_cgan
Copy-Item -Path "data\normal" -Destination "data_cgan\normal" -Recurse
Copy-Item -Path "data\alzheimer" -Destination "data_cgan\alzheimer" -Recurse

# Then run the training
python train.py --model cgan --data_dir data_cgan --epochs 100 --batch_size 32 --save_interval 10 --img_size 128
```

### Train CycleGAN
```
python train.py --model cyclegan --normal_dir data/normal --alzheimer_dir data/alzheimer --epochs 200 --batch_size 1 --save_interval 20
```

### Train Super-Resolution GAN
```
python train.py --model srgan --high_res_dir data/high_res --epochs 200 --batch_size 4 --save_interval 20 --scale_factor 4
```

## Project Structure
- `models/`: Contains the implementation of different GAN architectures
  - `conditional_gan.py`: Conditional GAN for generating brain images with/without Alzheimer's
  - `cycle_gan.py`: CycleGAN for transforming normal brain images to Alzheimer's and vice versa
  - `super_resolution_gan.py`: Super-Resolution GAN for enhancing low-resolution brain images
- `utils/`: Helper functions for data processing and visualization
  - `data_processing.py`: Functions for loading, processing, and augmenting data
  - `evaluation.py`: Functions for evaluating model performance
  - `download_data.py`: Utility for downloading datasets from Kaggle
- `app.py`: Streamlit interface entry point
- `train.py`: Script for training the different GAN models
- `download_dataset.py`: Script for downloading and organizing Alzheimer's datasets
- `main.py`: Complete workflow script (download, train, interface)
- `requirements.txt`: Required dependencies

## Generated Samples
Model outputs will be saved to:
- `generated_samples/`: Sample images generated during training
- `saved_models/`: Trained model weights

## Data Sources
The default dataset used is the [Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist/alzheimers-dataset-4-class-of-images) from Kaggle, which contains:
- NonDemented: Normal brain MRI scans
- VeryMildDemented, MildDemented, ModerateDemented: Different stages of Alzheimer's disease

You can specify a different Kaggle dataset using the `--dataset` parameter:
```
python download_dataset.py --dataset another_username/another_dataset
```

## Main Script Options

The `main.py` script provides a complete workflow and accepts the following arguments:

### Data Download Options
- `--kaggle_username`: Your Kaggle username
- `--kaggle_key`: Your Kaggle API key
- `--dataset`: Kaggle dataset identifier (default: tourist/alzheimers-dataset-4-class-of-images)
- `--output_dir`: Directory to save the organized dataset (default: data)

### Training Options
- `--epochs`: Number of epochs to train for (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--save_interval`: Interval for saving sample images (default: 10)
- `--scale_factor`: Scale factor for super-resolution (default: 4)

### Skip Options
- `--skip_download`: Skip dataset download step
- `--skip_training`: Skip model training step
- `--skip_interface`: Skip launching the Streamlit interface 