#!/usr/bin/env python3
"""
Main entry point for Alzheimer's Detection using GANs.
This script runs all components in sequence:
1. Download and organize dataset from Kaggle
2. Train the three GAN models (Conditional GAN, CycleGAN, Super-Resolution GAN)
3. Launch the Streamlit interface
"""

import os
import sys
import argparse
import subprocess
import time
import tensorflow as tf
import logging
import traceback
import pkg_resources
import site
import platform
import warnings

# Suppress TensorFlow warnings and set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
tf.get_logger().setLevel('ERROR')  # Suppress TF warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras.src.backend')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow.python.keras.losses')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow.python.ops.losses.losses_impl')

# Import TensorFlow with compatibility mode
tf.compat.v1.disable_eager_execution()  # Disable eager execution for better compatibility
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TF v1 logging

from utils.download_data import setup_kaggle_credentials, download_alzheimer_dataset

def setup_opencv_path():
    """Setup OpenCV path in the system"""
    try:
        # Get Python site-packages directory
        site_packages = site.getsitepackages()[0]
        
        # Add OpenCV path to environment variables
        if platform.system() == 'Windows':
            # Find the actual cv2 directory
            python_path = os.path.dirname(sys.executable)
            venv_path = os.path.dirname(os.path.dirname(sys.executable))  # Go up two levels from python.exe
            
            opencv_paths = [
                os.path.join(site_packages, 'cv2'),
                os.path.join(python_path, 'Lib', 'site-packages', 'cv2'),
                os.path.join(python_path, 'Lib', 'site-packages', 'opencv-python', 'cv2'),
                os.path.join(venv_path, 'Lib', 'site-packages', 'cv2'),
                os.path.join(venv_path, 'Lib', 'site-packages', 'opencv-python', 'cv2'),
                os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs', 'Python', 'Python3x', 'Lib', 'site-packages', 'cv2')
            ]
            
            # Try to find the actual cv2 directory
            opencv_path = None
            for path in opencv_paths:
                if os.path.exists(path):
                    opencv_path = path
                    print(f"Found OpenCV at: {path}")
                    break
            
            if opencv_path:
                # Add to PATH using PowerShell command
                ps_command = f'$env:PATH = "{opencv_path};" + $env:PATH'
                subprocess.run(['powershell', '-Command', ps_command], check=True)
                print(f"✓ Added OpenCV path: {opencv_path}")
                
                # Also set it in the current process
                os.environ['PATH'] = opencv_path + os.pathsep + os.environ['PATH']
                
                # Add the parent directory to Python path
                parent_dir = os.path.dirname(opencv_path)
                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)
                
                # Verify the path was added
                try:
                    import cv2
                    print("✓ OpenCV imported successfully")
                    print(f"OpenCV version: {cv2.__version__}")
                    return True
                except ImportError as e:
                    print(f"! OpenCV import failed after path setup: {str(e)}")
                    return False
            else:
                print("! OpenCV path not found in common locations")
                print("Searched in:")
                for path in opencv_paths:
                    print(f"  - {path}")
                return False
        else:
            # For Linux/Mac
            opencv_path = os.path.join(site_packages, 'cv2')
            if os.path.exists(opencv_path):
                os.environ['LD_LIBRARY_PATH'] = opencv_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
                print(f"✓ Added OpenCV path: {opencv_path}")
                return True
            else:
                print(f"! OpenCV path not found at: {opencv_path}")
                return False
        
    except Exception as e:
        print(f"! Error setting up OpenCV path: {str(e)}")
        return False

def verify_installation(package_name):
    """Verify if a package is installed correctly"""
    try:
        if package_name == "opencv-python":
            import cv2
            print(f"✓ OpenCV version {cv2.__version__} is installed and working")
            return True
        elif package_name == "scikit-learn":
            import sklearn
            print(f"✓ scikit-learn version {sklearn.__version__} is installed and working")
            return True
        else:
            __import__(package_name)
            print(f"✓ {package_name} is installed and working")
            return True
    except ImportError as e:
        print(f"✗ {package_name} import failed: {str(e)}")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        # Special handling for opencv-python
        if package_name == "opencv-python":
            # Try different versions in order of preference
            versions = [
                "opencv-python==4.5.5.64",  # Stable version
                "opencv-python==4.4.0.46",  # Alternative stable version
                "opencv-python==4.3.0.38",  # Another stable version
                "opencv-python==3.4.17.63",  # Legacy stable version
                "opencv-python-headless==4.5.5.64",  # Headless version
                "opencv-python-headless==4.4.0.46",  # Alternative headless version
                "opencv-python-headless==4.3.0.38",  # Another headless version
                "opencv-python-headless==3.4.17.63",  # Legacy headless version
                "opencv-python"  # Latest version as last resort
            ]
            
            # First uninstall any existing opencv
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
            except:
                pass
            
            # Try installing with specific options
            for version in versions:
                try:
                    print(f"Trying to install {version}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--no-cache-dir",
                        "--only-binary", ":all:",
                        "--no-deps",  # Don't install dependencies
                        version
                    ])
                    
                    # Now install dependencies separately
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install",
                            "--no-cache-dir",
                            "--only-binary", ":all:",
                            "numpy>=1.23.5,<2.0.0"
                        ])
                    except:
                        pass
                    
                    # Setup OpenCV path after installation
                    if setup_opencv_path():
                        print(f"✓ Successfully installed {version}")
                        return True
                    else:
                        print(f"! Installed {version} but path setup failed")
                        continue
                        
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {version}: {str(e)}")
                    continue
            
            print("All opencv-python installation attempts failed")
            return False
        
        # Special handling for scikit-learn
        elif package_name == "scikit-learn":
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir",
                    "--only-binary", ":all:",
                    "scikit-learn>=1.2.0"
                ])
                return True
            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--upgrade",
                        "--no-cache-dir",
                        "scikit-learn"
                    ])
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install scikit-learn: {str(e)}")
                    return False
        
        # For other packages
        else:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir",
                    "--only-binary", ":all:",
                    package_name
                ])
                return True
            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--upgrade",
                        "--no-cache-dir",
                        package_name
                    ])
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {package_name}: {str(e)}")
                    return False
    except Exception as e:
        print(f"Error installing {package_name}: {str(e)}")
        return False

def install_requirements():
    """Install all required packages from requirements.txt"""
    try:
        print("\nInstalling required packages...")
        # First try to upgrade pip itself
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install numpy first as it's a dependency for opencv
        try:
            # Install numpy 1.23.5 which is compatible with all dependencies
            print("\nInstalling numpy 1.23.5...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:", "numpy==1.23.5"])
            print("✓ Successfully installed numpy 1.23.5")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install numpy 1.23.5: {str(e)}")
            try:
                # Try the latest compatible version
                print("\nTrying to install latest compatible numpy version...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--only-binary", ":all:", "numpy>=1.23.5,<2.0.0"])
                print("✓ Successfully installed latest compatible numpy")
            except subprocess.CalledProcessError as e2:
                print(f"Failed to install numpy: {str(e2)}")
                print("Warning: Continuing with requirements installation...")
        
        # Install specific packages with their versions
        packages = [
            "seaborn>=0.12.0",
            "scikit-learn>=1.2.0",
            "tqdm>=4.65.0",
            "opencv-python>=4.5.5.64"
        ]
        
        for package in packages:
            try:
                print(f"\nInstalling {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir",
                    "--only-binary", ":all:",
                    package
                ])
                print(f"✓ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {str(e)}")
                # Try alternative installation method
                try:
                    print(f"Trying alternative installation method for {package}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--upgrade",
                        "--no-cache-dir",
                        package
                    ])
                    print(f"✓ Successfully installed {package} using alternative method")
                except subprocess.CalledProcessError as e2:
                    print(f"✗ Failed to install {package}: {str(e2)}")
                    return False
        
        # Then install any remaining requirements
        print("\nInstalling other requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "--only-binary", ":all:", "-r", "requirements.txt"])
        print("✓ Successfully installed all required packages")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {str(e)}")
        return False

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "="*80)
    print(f" {message} ".center(80, "="))
    print("="*80 + "\n")

def run_command(command, description):
    """Run a shell command and handle errors"""
    print_header(description)
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"Command completed with exit code: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def download_data(args):
    """Download and organize Alzheimer's dataset"""
    print_header("DOWNLOADING AND ORGANIZING DATASET")
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials(args.kaggle_username, args.kaggle_key):
        print("\nNo Kaggle credentials found. You have two options:")
        print("1. Run this script with Kaggle credentials:")
        print("   python main.py --kaggle_username YOUR_USERNAME --kaggle_key YOUR_KEY")
        print("\n2. Or set up credentials manually:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click on 'Create New API Token' to download kaggle.json")
        print("   - Move the downloaded file to ~/.kaggle/kaggle.json")
        print("   - Then run this script again without credentials parameters")
        
        if not args.skip_download:
            sys.exit(1)
        else:
            print("Skipping download as --skip_download was specified.")
            return False
    
    # Download and organize the dataset
    paths = download_alzheimer_dataset(args.output_dir, args.dataset)
    
    if not paths:
        if not args.skip_download:
            sys.exit(1)
        else:
            print("Skipping download as --skip_download was specified.")
            return False
    
    return True

def train_models(args):
    """Train all three GAN models"""
    if args.skip_training:
        print_header("SKIPPING MODEL TRAINING (--skip_training specified)")
        return True
    
    # Check if data directories exist
    normal_dir = os.path.join(args.output_dir, 'normal')
    alzheimer_dir = os.path.join(args.output_dir, 'alzheimer')
    high_res_dir = os.path.join(args.output_dir, 'high_res')
    
    if not all(os.path.exists(d) for d in [normal_dir, alzheimer_dir, high_res_dir]):
        print("Error: One or more data directories are missing. Please run data download first.")
        return False
    
    # Train Conditional GAN
    cmd = f"python train.py --model cgan --data_dir {args.output_dir} --epochs {args.epochs} " \
          f"--batch_size {args.batch_size} --save_interval {args.save_interval}"
    if not run_command(cmd, "TRAINING CONDITIONAL GAN"):
        return False
    
    # Train CycleGAN
    cmd = f"python train.py --model cyclegan --normal_dir {normal_dir} " \
          f"--alzheimer_dir {alzheimer_dir} --epochs {args.epochs} " \
          f"--batch_size 1 --save_interval {args.save_interval}"
    if not run_command(cmd, "TRAINING CYCLEGAN"):
        return False
    
    # Train SRGAN
    cmd = f"python train.py --model srgan --high_res_dir {high_res_dir} " \
          f"--epochs {args.epochs} --batch_size 4 " \
          f"--save_interval {args.save_interval} --scale_factor {args.scale_factor}"
    if not run_command(cmd, "TRAINING SUPER-RESOLUTION GAN"):
        return False
    
    return True

def launch_interface():
    """Launch the Streamlit interface"""
    print_header("LAUNCHING STREAMLIT INTERFACE")
    cmd = "streamlit run app.py"
    run_command(cmd, "STREAMLIT INTERFACE")

def check_dependencies():
    """Check if all required packages are installed and install missing ones"""
    required_packages = {
        "tensorflow": "Deep learning framework",
        "numpy": "Numerical computations",
        "PIL": "Image processing",
        "matplotlib": "Visualization",
        "streamlit": "Web interface",
        "opencv-python": "Image processing",
        "scikit-learn": "Machine learning utilities",
        "seaborn": "Statistical visualization",
        "tqdm": "Progress bars",
        "pandas": "Data manipulation",
        "keras": "Deep learning framework"
    }
    
    print("\nChecking dependencies...")
    missing_packages = []
    for package, description in required_packages.items():
        try:
            if package == "PIL":
                import PIL
                print(f"✓ PIL (Pillow) is installed")
            elif package == "opencv-python":
                import cv2
                print(f"✓ OpenCV version {cv2.__version__} is installed")
            elif package == "scikit-learn":
                import sklearn
                print(f"✓ scikit-learn version {sklearn.__version__} is installed")
            else:
                __import__(package)
                print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append((package, description))
            print(f"✗ {package} is missing ({description})")
    
    if missing_packages:
        print("\nInstalling missing packages...")
        for package, description in missing_packages:
            print(f"\nInstalling {package} ({description})...")
            if not install_package(package):
                print(f"\nFailed to install {package}. Please install it manually using:")
                print(f"pip install {package}")
                return False
            
            # Verify installation
            try:
                if package == "PIL":
                    import PIL
                    print(f"✓ PIL (Pillow) is now installed")
                elif package == "opencv-python":
                    import cv2
                    print(f"✓ OpenCV version {cv2.__version__} is now installed")
                elif package == "scikit-learn":
                    import sklearn
                    print(f"✓ scikit-learn version {sklearn.__version__} is now installed")
                else:
                    __import__(package)
                    print(f"✓ {package} is now installed")
            except ImportError:
                print(f"✗ Failed to install {package}")
                return False
    
    print("\nAll dependencies are installed!")
    return True

def create_data_directories():
    """Create required data directories if they don't exist"""
    required_dirs = [
        "data/normal",
        "data/alzheimer",
        "data/NonDemented",
        "data/VeryMildDemented",
        "data/MildDemented",
        "data/ModerateDemented"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"✗ Failed to create directory {dir_path}: {str(e)}")
                return False
    return True

def check_data_directories():
    """Check if required data directories exist"""
    required_dirs = [
        "data/normal",
        "data/alzheimer",
        "data/NonDemented",
        "data/VeryMildDemented",
        "data/MildDemented",
        "data/ModerateDemented"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            missing_dirs.append(dir_path)
            print(f"✗ Missing directory: {dir_path}")
    
    if missing_dirs:
        print("\nCreating missing directories...")
        if not create_data_directories():
            print("\nFailed to create some directories. Please create them manually.")
            return False
    
    return True

def main():
    try:
        print_header("Alzheimer's Detection System")
        print("\nChecking system requirements...")
        
        # Check Python version
        print(f"\nPython version: {sys.version}")
        
        # First try to install all requirements
        if not install_requirements():
            print("\nFalling back to individual package installation...")
        
        # Check and install dependencies
        if not check_dependencies():
            print("\nSome packages could not be installed. Please install them manually.")
            return
        
        # Check and create data directories
        if not check_data_directories():
            print("\nSome directories could not be created. Please create them manually.")
            return
        
        print("\nAll requirements satisfied!")
        
        # Setup OpenCV path
        if not setup_opencv_path():
            print("\nWarning: OpenCV path setup failed. Some features may not work correctly.")
        
        while True:
            print("\nChoose an option:")
            print("1. Run Streamlit Web Interface")
            print("2. Run Alzheimer's Classifier")
            print("3. Train Conditional GAN")
            print("4. Train CycleGAN")
            print("5. Train Super-Resolution GAN")
            print("6. Exit")
            
            try:
                choice = input("\nEnter your choice (1-6): ")
                
                if choice == "1":
                    print("\nStarting Streamlit web interface...")
                    os.system("streamlit run app.py")
                
                elif choice == "2":
                    print("\nRunning Alzheimer's Classifier...")
                    os.system("python run_classifier.py")
                
                elif choice == "3":
                    print("\nTraining Conditional GAN...")
                    os.system("python train.py --model cgan --data_dir data_cgan --epochs 200 --batch_size 32 --save_interval 10 --img_size 128")
                
                elif choice == "4":
                    print("\nTraining CycleGAN...")
                    os.system("python train.py --model cyclegan --data_dir data_cyclegan --epochs 200 --batch_size 8 --save_interval 10 --img_size 128")
                
                elif choice == "5":
                    print("\nTraining Super-Resolution GAN...")
                    os.system("python train.py --model srgan --data_dir data_srgan --epochs 200 --batch_size 16 --save_interval 10 --img_size 128")
                
                elif choice == "6":
                    print("\nExiting...")
                    sys.exit(0)
                
                else:
                    print("\nInvalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                continue
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("\nTraceback:")
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"\nA critical error occurred: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 