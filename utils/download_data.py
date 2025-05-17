import os
import subprocess
import zipfile
import argparse
import shutil
import sys
from tqdm import tqdm

def setup_kaggle_credentials(kaggle_username=None, kaggle_key=None):
    """
    Setup Kaggle credentials for API access.
    
    Args:
        kaggle_username: Kaggle username
        kaggle_key: Kaggle API key
    
    Returns:
        bool: True if credentials are set up successfully, False otherwise
    """
    # Check if credentials are already set in environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        print("Kaggle credentials already set in environment variables.")
        return True
        
    # Check if credentials file exists
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_file):
        print(f"Kaggle credentials found at {kaggle_file}")
        # Set permissions to avoid warnings
        try:
            os.chmod(kaggle_file, 0o600)
        except:
            pass
        return True
    
    # If username and key are provided, create the credentials file
    if kaggle_username and kaggle_key:
        os.makedirs(kaggle_dir, exist_ok=True)
        
        import json
        with open(kaggle_file, 'w') as f:
            json.dump({
                "username": kaggle_username,
                "key": kaggle_key
            }, f)
        
        # Set permissions
        try:
            os.chmod(kaggle_file, 0o600)
        except:
            pass
        
        print(f"Kaggle credentials saved to {kaggle_file}")
        return True
    
    print("No Kaggle credentials found. Please provide your Kaggle username and API key.")
    print("You can find your API key at https://www.kaggle.com/account")
    return False

def download_dataset(dataset, output_dir, unzip=True):
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset: Kaggle dataset identifier (username/dataset-name)
        output_dir: Directory to save the dataset
        unzip: Whether to unzip the downloaded file
        
    Returns:
        str: Path to the downloaded/extracted dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if kaggle CLI is installed
    try:
        subprocess.check_output(['kaggle', '--version'])
    except:
        print("Kaggle CLI not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
    
    # Download the dataset
    print(f"Downloading {dataset} to {output_dir}...")
    try:
        subprocess.check_call(['kaggle', 'datasets', 'download', '-d', dataset, '-p', output_dir])
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and Kaggle credentials.")
        return None
    
    # Find the downloaded zip file
    zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
    if not zip_files:
        print("No zip file found after download.")
        return output_dir
    
    zip_path = os.path.join(output_dir, zip_files[0])
    
    if unzip:
        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print("Extraction complete!")
            
            # Optionally remove the zip file after extraction
            os.remove(zip_path)
            print(f"Removed {zip_path}")
        except Exception as e:
            print(f"Error extracting zip file: {e}")
    
    return output_dir

def organize_alzheimer_dataset(dataset_dir, output_dir):
    """
    Organize Alzheimer's dataset into the structure required by the training script.
    
    Args:
        dataset_dir: Directory containing the downloaded dataset
        output_dir: Directory to organize the data into
        
    Returns:
        dict: Paths to the organized data directories
    """
    # Create output directories
    normal_dir = os.path.join(output_dir, 'normal')
    alzheimer_dir = os.path.join(output_dir, 'alzheimer')
    high_res_dir = os.path.join(output_dir, 'high_res')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(alzheimer_dir, exist_ok=True)
    os.makedirs(high_res_dir, exist_ok=True)
    
    # Process the dataset based on common Alzheimer's dataset structures
    # This is an example for the Alzheimer's Dataset (4 class of Images)
    alzheimer_class_dirs = {
        'NonDemented': normal_dir,
        'VeryMildDemented': alzheimer_dir,
        'MildDemented': alzheimer_dir,
        'ModerateDemented': alzheimer_dir
    }
    
    # Check if this is the common Alzheimer's dataset structure
    for class_name in alzheimer_class_dirs.keys():
        potential_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(potential_dir):
            print(f"Found directory: {class_name}")
            # Copy the files to the appropriate directory
            files = [f for f in os.listdir(potential_dir) if os.path.isfile(os.path.join(potential_dir, f))]
            
            for file in tqdm(files, desc=f"Copying {class_name} files"):
                src_path = os.path.join(potential_dir, file)
                dst_path = os.path.join(alzheimer_class_dirs[class_name], file)
                # Copy the file
                shutil.copy2(src_path, dst_path)
                
                # Also copy non-demented files to high_res for SRGAN training
                if class_name == 'NonDemented':
                    high_res_path = os.path.join(high_res_dir, file)
                    shutil.copy2(src_path, high_res_path)
            
    # Return the paths
    return {
        'data_dir': output_dir,
        'normal_dir': normal_dir,
        'alzheimer_dir': alzheimer_dir,
        'high_res_dir': high_res_dir
    }

def download_alzheimer_dataset(output_dir='data', dataset=None):
    """
    Download and organize an Alzheimer's dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the organized dataset
        dataset: Kaggle dataset identifier (default: popular Alzheimer's dataset)
        
    Returns:
        dict: Paths to the organized data directories
    """
    # Default to a popular Alzheimer's dataset if none is specified
    if dataset is None:
        dataset = "tourist/alzheimers-dataset-4-class-of-images"
    
    # Create a temporary directory for the raw download
    temp_dir = os.path.join(output_dir, 'temp_download')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download the dataset
    download_path = download_dataset(dataset, temp_dir)
    
    if not download_path:
        return None
    
    # Organize the dataset
    organized_paths = organize_alzheimer_dataset(download_path, output_dir)
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    
    # Print summary
    print("\nDataset organization complete!")
    print(f"Normal brain images: {len(os.listdir(organized_paths['normal_dir']))} files")
    print(f"Alzheimer's brain images: {len(os.listdir(organized_paths['alzheimer_dir']))} files")
    print(f"High-resolution images for SRGAN: {len(os.listdir(organized_paths['high_res_dir']))} files")
    
    return organized_paths

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Download and organize Alzheimer's datasets from Kaggle")
    parser.add_argument('--kaggle_username', type=str, help='Your Kaggle username')
    parser.add_argument('--kaggle_key', type=str, help='Your Kaggle API key')
    parser.add_argument('--dataset', type=str, default="tourist/alzheimers-dataset-4-class-of-images",
                        help='Kaggle dataset identifier (username/dataset-name)')
    parser.add_argument('--output_dir', type=str, default="data",
                        help='Directory to save the organized dataset')
    
    args = parser.parse_args()
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials(args.kaggle_username, args.kaggle_key):
        print("Failed to set up Kaggle credentials. Exiting...")
        sys.exit(1)
    
    # Download and organize the dataset
    download_alzheimer_dataset(args.output_dir, args.dataset) 