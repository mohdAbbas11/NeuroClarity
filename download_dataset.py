#!/usr/bin/env python3
"""
Script to download and organize Alzheimer's datasets from Kaggle.
This is a wrapper around utils/download_data.py with a simpler interface.
"""

import argparse
import sys
from utils.download_data import setup_kaggle_credentials, download_alzheimer_dataset

def main():
    parser = argparse.ArgumentParser(description="Download and organize Alzheimer's datasets from Kaggle")
    parser.add_argument('--kaggle_username', type=str, help='Your Kaggle username')
    parser.add_argument('--kaggle_key', type=str, help='Your Kaggle API key')
    parser.add_argument('--dataset', type=str, 
                        help='Kaggle dataset identifier (username/dataset-name). Default: tourist/alzheimers-dataset-4-class-of-images')
    parser.add_argument('--output_dir', type=str, default="data",
                        help='Directory to save the organized dataset')
    
    args = parser.parse_args()
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials(args.kaggle_username, args.kaggle_key):
        print("\nNo Kaggle credentials found. You have two options:")
        print("1. Run this script with your Kaggle credentials:")
        print("   python download_dataset.py --kaggle_username YOUR_USERNAME --kaggle_key YOUR_KEY")
        print("\n2. Or set up credentials manually:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click on 'Create New API Token' to download kaggle.json")
        print("   - Move the downloaded file to ~/.kaggle/kaggle.json")
        print("   - Then run this script again without credentials parameters")
        sys.exit(1)
    
    # Download and organize the dataset
    download_alzheimer_dataset(args.output_dir, args.dataset)
    
    print("\nDownload complete! You can now train your models with:")
    print("1. Conditional GAN:")
    print(f"   python train.py --model cgan --data_dir {args.output_dir}")
    print("\n2. CycleGAN:")
    print(f"   python train.py --model cyclegan --normal_dir {args.output_dir}/normal --alzheimer_dir {args.output_dir}/alzheimer")
    print("\n3. Super-Resolution GAN:")
    print(f"   python train.py --model srgan --high_res_dir {args.output_dir}/high_res")
    print("\nOr run the Streamlit interface:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main() 