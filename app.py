import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array, to_categorical
from keras.models import load_model
import cv2
from PIL import Image
import io

# Configure GPU memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU found: {physical_devices}")
else:
    print("No GPU found. Running on CPU.")

# Check for required packages
def check_dependencies():
    missing_packages = []
    
    required_packages = {
        "seaborn": "Enhanced visualizations",
        "scikit-learn": "Metrics and evaluation tools",
        "tqdm": "Progress tracking",
        "opencv-python": "Image processing"
    }
    
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append((package, description))
    
    return missing_packages

# Check for model files and data directories
def check_resources():
    missing_resources = []
    
    # Check data directories
    data_dirs = {
        "Normal brain images": "data/normal",
        "Alzheimer's brain images": "data/alzheimer",
        "High-resolution images": "data/high_res"
    }
    
    for name, dir_path in data_dirs.items():
        if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
            missing_resources.append(name)
    
    # Check model files
    model_files = {
        "Conditional GAN": "models/conditional_gan_model.h5",
        "CycleGAN A2B": "models/cyclegan_generator_A2B.h5",
        "CycleGAN B2A": "models/cyclegan_generator_B2A.h5",
        "Super-Resolution GAN": "models/srgan_generator.h5"
    }
    
    for name, file_path in model_files.items():
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:  # Check if placeholder or missing
            missing_resources.append(name)
    
    return missing_resources

# Import models
from models.conditional_gan import ConditionalGAN
from models.cycle_gan import CycleGAN
from models.super_resolution_gan import SRGAN
from models.alzheimer_classifier import AlzheimerClassifier

# Import utils
from utils.data_processing import visualize_results, generate_low_res_images
from utils.evaluation import evaluate_image_quality

# Set page configuration
st.set_page_config(
    page_title="Alzheimer's Detection using GANs",
    page_icon="🧠",
    layout="wide"
)

# Function to preprocess images
def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        # Handle numpy array input
        img = cv2.resize(image, target_size)
    else:
        # Handle uploaded file input
        img = image.resize(target_size)
        img = np.array(img)
    
    # Ensure image has 3 channels
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # Normalize to [-1, 1]
    img = img / 127.5 - 1
    
    return img

def load_saved_model(model_type):
    """Load a saved model based on type"""
    models_dir = "saved_models"
    
    if not os.path.exists(models_dir):
        st.warning(f"No saved models found in '{models_dir}'. Please train models first.")
        return None
    
    try:
        if model_type == "conditional_gan":
            generator_path = os.path.join(models_dir, "cgan_generator.h5")
            discriminator_path = os.path.join(models_dir, "cgan_discriminator.h5")
            
            if os.path.exists(generator_path) and os.path.exists(discriminator_path):
                model = ConditionalGAN()
                model.load_models(generator_path, discriminator_path)
                return model
        
        elif model_type == "cycle_gan":
            g_AB_path = os.path.join(models_dir, "cyclegan_g_AB.h5")
            g_BA_path = os.path.join(models_dir, "cyclegan_g_BA.h5")
            d_A_path = os.path.join(models_dir, "cyclegan_d_A.h5")
            d_B_path = os.path.join(models_dir, "cyclegan_d_B.h5")
            
            if os.path.exists(g_AB_path) and os.path.exists(g_BA_path) and \
               os.path.exists(d_A_path) and os.path.exists(d_B_path):
                model = CycleGAN()
                model.load_models(g_AB_path, g_BA_path, d_A_path, d_B_path)
                return model
        
        elif model_type == "srgan":
            generator_path = os.path.join(models_dir, "srgan_generator.h5")
            discriminator_path = os.path.join(models_dir, "srgan_discriminator.h5")
            
            if os.path.exists(generator_path) and os.path.exists(discriminator_path):
                model = SRGAN()
                model.load_models(generator_path, discriminator_path)
                return model
        
        st.warning(f"Model files for {model_type} not found in '{models_dir}'.")
        return None
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_sample_images():
    """Load sample images for demo purposes"""
    # In a real application, these would be actual brain scan images
    # For demo purposes, we're creating placeholder images
    normal_img = np.ones((128, 128, 3)) * 0.8
    # Add some patterns to make it look brain-like
    for i in range(20, 100):
        for j in range(30, 90):
            if (i - 60) ** 2 + (j - 60) ** 2 < 800:
                normal_img[i, j] = 0.5
    
    alzheimer_img = np.ones((128, 128, 3)) * 0.7
    # Add different patterns for Alzheimer's
    for i in range(20, 100):
        for j in range(30, 90):
            if (i - 60) ** 2 + (j - 60) ** 2 < 600:
                alzheimer_img[i, j] = 0.3
            if (i - 80) ** 2 + (j - 40) ** 2 < 100:
                alzheimer_img[i, j] = 0.1
    
    # Normalize to [0, 1] for display
    normal_img = np.clip(normal_img, 0, 1)
    alzheimer_img = np.clip(alzheimer_img, 0, 1)
    
    return normal_img, alzheimer_img

def plot_to_image():
    """Convert matplotlib plot to image"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

# Sidebar
st.sidebar.title("Alzheimer's Detection using GANs")
app_mode = st.sidebar.selectbox(
    "Choose the GAN model",
    ["Home", "Conditional GAN", "CycleGAN", "Super-Resolution GAN", "Alzheimer Classifier"]
)

# Main page
if app_mode == "Home":
    st.title("Alzheimer's Detection using GANs")
    
    # Check for missing dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        st.warning("⚠️ Some dependencies are missing. The app will try to run with limited functionality.")
        with st.expander("See missing dependencies"):
            st.write("To install all dependencies, run: ```pip install seaborn scikit-learn tqdm opencv-python```")
            for package, description in missing_packages:
                st.write(f"- **{package}**: {description}")
    
    # Check for missing resources
    missing_resources = check_resources()
    if missing_resources:
        st.info("ℹ️ Using placeholder data and models for demonstration")
        with st.expander("See missing resources"):
            st.write("The following resources are using placeholders:")
            for resource in missing_resources:
                st.write(f"- {resource}")
            
            st.write("You can generate sample data by running: ```python prepare_data.py```")
    
    st.markdown("""
        This application uses different GAN architectures for Alzheimer's disease detection and visualization:
        
        1. **Conditional GAN**: Generate brain images conditioned on being normal or having Alzheimer's disease
        2. **CycleGAN**: Transform normal brain images to Alzheimer's disease and vice versa
        3. **Super-Resolution GAN**: Enhance low-resolution brain images for better diagnosis
        
        Select a model from the sidebar to begin.
    """)
    
    st.image("https://www.radiologyinfo.org/gallery-items/images/brain-mri.jpg", caption="Sample brain MRI image", width=400)
    
    st.subheader("How to use this app")
    st.markdown("""
        - Upload your own brain scan images or use sample images
        - Apply different GAN models to detect and visualize Alzheimer's disease
        - Compare original and generated images
        - Generate synthetic data for research purposes
    """)

elif app_mode == "Conditional GAN":
    st.title("Conditional GAN for Alzheimer's Detection")
    st.markdown("""
        Conditional GANs can generate brain images conditioned on class labels (normal vs. Alzheimer's).
        This can be useful for:
        - Data augmentation for research studies
        - Visualizing disease progression
        - Generating synthetic examples for training other models
    """)
    
    # Load model (or use placeholder functionality if model is not available)
    model = load_saved_model("conditional_gan")
    
    option = st.radio("Select option:", ["Generate random samples", "Use specific condition"])
    
    if option == "Generate random samples":
        if st.button("Generate samples"):
            # Create placeholder data if model is not available
            if model is None:
                st.warning("Using placeholder data since model is not loaded.")
                normal_img, alzheimer_img = load_sample_images()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(normal_img, caption="Generated normal brain", width=300)
                with col2:
                    st.image(alzheimer_img, caption="Generated Alzheimer's brain", width=300)
            else:
                # Generate samples with the actual model
                labels = to_categorical([0, 1], 2)  # 0=Normal, 1=Alzheimer's
                generated_images = model.generate_samples(n_samples=2, labels=labels)
                
                # Convert to display format
                gen_display = []
                for img in generated_images:
                    # Convert tensor to numpy array if needed
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    # Convert from [-1,1] to [0,1] range
                    img = (img + 1) / 2
                    # Convert to uint8 for display
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    gen_display.append(img)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(gen_display[0], caption="Generated normal brain", width=300)
                with col2:
                    st.image(gen_display[1], caption="Generated Alzheimer's brain", width=300)
    
    else:  # Use specific condition
        condition = st.radio("Select condition to generate:", ["Normal", "Alzheimer's"])
        
        if st.button("Generate"):
            # Create placeholder data if model is not available
            if model is None:
                st.warning("Using placeholder data since model is not loaded.")
                normal_img, alzheimer_img = load_sample_images()
                
                if condition == "Normal":
                    st.image(normal_img, caption=f"Generated {condition} brain", width=300)
                else:
                    st.image(alzheimer_img, caption=f"Generated {condition} brain", width=300)
            else:
                # Generate with the actual model
                label_idx = 0 if condition == "Normal" else 1
                label = to_categorical([label_idx], 2)
                
                generated_image = model.generate_samples(n_samples=1, labels=label)[0]
                # Convert tensor to numpy array if needed
                if hasattr(generated_image, 'numpy'):
                    generated_image = generated_image.numpy()
                # Convert from [-1,1] to [0,1] range
                gen_display = (generated_image + 1) / 2
                # Convert to uint8 for display
                gen_display = (gen_display * 255).clip(0, 255).astype(np.uint8)
                
                st.image(gen_display, caption=f"Generated {condition} brain", width=300)

elif app_mode == "CycleGAN":
    st.title("CycleGAN for Alzheimer's Image Translation")
    st.markdown("""
        CycleGAN can transform images between normal and Alzheimer's domains without paired data.
        This is useful for:
        - Understanding disease characteristics
        - Simulating disease progression
        - Converting between normal and Alzheimer's domains for diagnostic purposes
    """)
    
    model = load_saved_model("cycle_gan")
    
    upload_option = st.radio("Choose image source:", ["Upload image", "Use sample image"])
    
    if upload_option == "Upload image":
        uploaded_file = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            translate_option = st.radio("Translate to:", ["Normal to Alzheimer's", "Alzheimer's to Normal"])
            
            if st.button("Translate Image"):
                # Preprocess the image
                preprocessed_img = preprocess_image(image)
                preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
                
                # Create placeholder data if model is not available
                if model is None:
                    st.warning("Using placeholder data since model is not loaded.")
                    normal_img, alzheimer_img = load_sample_images()
                    
                    if translate_option == "Normal to Alzheimer's":
                        result_img = alzheimer_img
                    else:
                        result_img = normal_img
                    
                    st.image(result_img, caption="Translated image (placeholder)", width=300)
                else:
                    # Translate with the actual model
                    if translate_option == "Normal to Alzheimer's":
                        translated_img = model.generate_alzheimer_images(preprocessed_img)[0]
                    else:
                        translated_img = model.generate_normal_images(preprocessed_img)[0]
                    
                    # Convert to display format
                    translated_display = (translated_img + 1) / 2
                    
                    st.image(translated_display, caption="Translated image", width=300)
    else:
        # Use sample images
        st.write("Using sample brain images")
        normal_img, alzheimer_img = load_sample_images()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(normal_img, caption="Sample normal brain", width=300)
            use_normal = st.button("Use normal brain image")
        
        with col2:
            st.image(alzheimer_img, caption="Sample Alzheimer's brain", width=300)
            use_alzheimer = st.button("Use Alzheimer's brain image")
        
        if use_normal or use_alzheimer:
            source_img = normal_img if use_normal else alzheimer_img
            source_type = "Normal" if use_normal else "Alzheimer's"
            target_type = "Alzheimer's" if use_normal else "Normal"
            
            # Create placeholder result if model is not available
            if model is None:
                st.warning("Using placeholder data since model is not loaded.")
                result_img = alzheimer_img if use_normal else normal_img
                
                st.write(f"Translation from {source_type} to {target_type}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(source_img, caption=f"Original {source_type}", width=300)
                with col2:
                    st.image(result_img, caption=f"Translated to {target_type}", width=300)
            else:
                # Translate with the actual model
                # Preprocess image
                preprocessed_img = preprocess_image(source_img)
                preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
                
                if use_normal:
                    translated_img = model.generate_alzheimer_images(preprocessed_img)[0]
                else:
                    translated_img = model.generate_normal_images(preprocessed_img)[0]
                
                # Convert to display format
                translated_display = (translated_img + 1) / 2
                
                st.write(f"Translation from {source_type} to {target_type}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(source_img, caption=f"Original {source_type}", width=300)
                with col2:
                    st.image(translated_display, caption=f"Translated to {target_type}", width=300)

elif app_mode == "Super-Resolution GAN":
    st.title("Super-Resolution GAN for Alzheimer's Imaging")
    st.markdown("""
        Super-Resolution GAN enhances low-resolution brain images to high-resolution.
        This is particularly useful for:
        - Improving diagnostic quality of low-resolution scans
        - Enhancing subtle features that may indicate Alzheimer's disease
        - Making older or lower quality scans more useful for diagnosis
    """)
    
    model = load_saved_model("srgan")
    
    upload_option = st.radio("Choose image source:", ["Upload image", "Use sample image"])
    
    if upload_option == "Upload image":
        uploaded_file = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            if st.button("Enhance Image"):
                # Create low-res version of the uploaded image
                high_res_img = np.array(image)
                
                # Ensure image is 3 channels
                if len(high_res_img.shape) == 2:
                    high_res_img = np.stack((high_res_img,)*3, axis=-1)
                elif high_res_img.shape[2] == 4:
                    high_res_img = high_res_img[:, :, :3]
                
                # Resize to target
                high_res_img = cv2.resize(high_res_img, (128, 128))
                
                # Create low-res version
                scale_factor = 4
                low_res_h, low_res_w = 128 // scale_factor, 128 // scale_factor
                low_res_img = cv2.resize(high_res_img, (low_res_w, low_res_h), interpolation=cv2.INTER_CUBIC)
                
                # Create bicubic upsampled version for comparison
                bicubic_img = cv2.resize(low_res_img, (128, 128), interpolation=cv2.INTER_CUBIC)
                
                # Create placeholder result if model is not available
                if model is None:
                    st.warning("Using placeholder data since model is not loaded.")
                    
                    # Use bicubic upsampling as the placeholder result with slight enhancement
                    enhanced_img = bicubic_img.copy()
                    enhanced_img = cv2.detailEnhance(enhanced_img, sigma_s=10, sigma_r=0.15)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(low_res_img, caption="Low Resolution", width=250)
                    with col2:
                        st.image(bicubic_img, caption="Bicubic Upsampling", width=250)
                    with col3:
                        st.image(enhanced_img, caption="SR-GAN Enhanced (simulated)", width=250)
                else:
                    # Normalize low-res image for the model
                    lr_input = low_res_img / 127.5 - 1
                    lr_input = np.expand_dims(lr_input, axis=0)
                    
                    # Generate high-res image
                    sr_img = model.generate_high_res(lr_input)[0]
                    
                    # Convert to display format
                    sr_display = (sr_img + 1) / 2
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(low_res_img, caption="Low Resolution", width=250)
                    with col2:
                        st.image(bicubic_img, caption="Bicubic Upsampling", width=250)
                    with col3:
                        st.image(sr_display, caption="SR-GAN Enhanced", width=250)
    else:
        # Use sample image
        st.write("Using sample brain image")
        normal_img, _ = load_sample_images()
        
        st.image(normal_img, caption="Sample brain image", width=300)
        
        if st.button("Enhance Sample Image"):
            # Create low-res version
            scale_factor = 4
            high_res_img = normal_img
            low_res_h, low_res_w = 128 // scale_factor, 128 // scale_factor
            low_res_img = cv2.resize(high_res_img, (low_res_w, low_res_h), interpolation=cv2.INTER_CUBIC)
            
            # Create bicubic upsampled version for comparison
            bicubic_img = cv2.resize(low_res_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            
            # Create placeholder result if model is not available
            if model is None:
                st.warning("Using placeholder data since model is not loaded.")
                
                # Use bicubic upsampling as the placeholder result with slight enhancement
                enhanced_img = bicubic_img.copy()
                enhanced_img = cv2.detailEnhance(enhanced_img.astype(np.uint8), sigma_s=10, sigma_r=0.15)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(low_res_img, caption="Low Resolution", width=250)
                with col2:
                    st.image(bicubic_img, caption="Bicubic Upsampling", width=250)
                with col3:
                    st.image(enhanced_img, caption="SR-GAN Enhanced (simulated)", width=250)
            else:
                # Normalize low-res image for the model
                lr_input = low_res_img / 127.5 - 1
                lr_input = np.expand_dims(lr_input, axis=0)
                
                # Generate high-res image
                sr_img = model.generate_high_res(lr_input)[0]
                
                # Convert to display format
                sr_display = (sr_img + 1) / 2
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(low_res_img, caption="Low Resolution", width=250)
                with col2:
                    st.image(bicubic_img, caption="Bicubic Upsampling", width=250)
                with col3:
                    st.image(sr_display, caption="SR-GAN Enhanced", width=250)

elif app_mode == "Alzheimer Classifier":
    st.title("Alzheimer's Disease Stage Classification")
    
    st.markdown("""
        This section uses a deep learning classifier to detect different stages of Alzheimer's disease:
        - **Non-Demented**: Normal brain without signs of Alzheimer's
        - **Very Mild Demented**: Early signs of Alzheimer's
        - **Mild Demented**: Moderate progression of Alzheimer's
        - **Moderate Demented**: Advanced stage of Alzheimer's
        
        The model was trained on brain MRI scans and can classify a new scan into one of these four categories.
    """)
    
    # Load the classifier model
    @st.cache_resource
    def load_classifier_model():
        model_path = "results/alzheimer_classifier_model.h5"
        if os.path.exists(model_path):
            try:
                classifier = AlzheimerClassifier(img_shape=(128, 128, 3), num_classes=4)
                classifier.load_model(model_path)
                return classifier
            except Exception as e:
                st.error(f"Error loading classifier model: {e}")
                return None
        else:
            st.warning(f"Model file not found at {model_path}. Using a placeholder model.")
            classifier = AlzheimerClassifier(img_shape=(128, 128, 3), num_classes=4)
            return classifier
    
    classifier = load_classifier_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])
    
    # Sample images selector
    st.subheader("Or select a sample image:")
    
    # Dynamically find sample images if they exist
    sample_images = []
    sample_dirs = ["data/NonDemented", "data/VeryMildDemented", "data/MildDemented", "data/ModerateDemented"]
    
    for dir_path in sample_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                sample_images.append(os.path.join(dir_path, files[0]))
    
    # Fallback to alternative directories if needed
    if not sample_images:
        alt_dirs = ["data/normal", "data/alzheimer"]
        for dir_path in alt_dirs:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    sample_images.append(os.path.join(dir_path, files[0]))
    
    # Create a grid of sample images or message if none available
    if sample_images:
        cols = st.columns(min(4, len(sample_images)))
        selected_sample = None
        
        for i, sample_path in enumerate(sample_images[:4]):  # Limit to 4 samples
            with cols[i]:
                img = Image.open(sample_path)
                st.image(img, width=150)
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    selected_sample = sample_path
        
        if selected_sample:
            st.success(f"Selected sample image: {os.path.basename(selected_sample)}")
            # Use the selected sample instead of uploaded file
            uploaded_file = selected_sample
    else:
        st.info("No sample images available. Please upload your own image.")
    
    # Process image if provided
    if uploaded_file is not None:
        try:
            # Load and display the image
            if isinstance(uploaded_file, str):
                # It's a file path
                image = Image.open(uploaded_file)
                image_path = uploaded_file
            else:
                # It's an uploaded file
                image = Image.open(uploaded_file)
                # Save to temporary file
                image_path = "temp_upload.jpg"
                image.save(image_path)
            
            st.image(image, caption="Uploaded MRI Scan", width=300)
            
            # Preprocess image
            processed_img = AlzheimerClassifier.preprocess_image(image_path, target_size=(128, 128))
            
            # Make prediction
            with st.spinner("Analyzing the brain scan..."):
                if classifier is not None:
                    prediction = classifier.predict_class(processed_img)
                    
                    # Display results
                    st.subheader("Classification Results")
                    
                    # Display class with confidence
                    class_name = prediction['class_name']
                    confidence = prediction['probabilities'][prediction['class_index']] * 100
                    
                    # Color code based on severity
                    if class_name == "NonDemented":
                        st.success(f"**Prediction: {class_name}** (Confidence: {confidence:.1f}%)")
                    elif class_name == "VeryMildDemented":
                        st.warning(f"**Prediction: {class_name}** (Confidence: {confidence:.1f}%)")
                    elif class_name == "MildDemented":
                        st.warning(f"**Prediction: {class_name}** (Confidence: {confidence:.1f}%)", icon="⚠️")
                    else:  # ModerateDemented
                        st.error(f"**Prediction: {class_name}** (Confidence: {confidence:.1f}%)")
                    
                    # Plot probabilities
                    st.subheader("Class Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
                    probs = [prediction['probabilities'][i] for i in range(4)]
                    
                    colors = ['green', 'yellow', 'orange', 'red']
                    ax.barh(class_names, probs, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Alzheimer\'s Disease Stage Probabilities')
                    
                    st.pyplot(fig)
                    
                    # Add interpretations
                    st.subheader("Interpretation")
                    if class_name == "NonDemented":
                        st.write("The scan appears to show a normal brain without significant signs of Alzheimer's disease.")
                    elif class_name == "VeryMildDemented":
                        st.write("The scan shows early signs that may be consistent with the beginning stages of Alzheimer's disease.")
                    elif class_name == "MildDemented":
                        st.write("The scan shows patterns consistent with mild Alzheimer's disease, including some brain tissue changes.")
                    else:  # ModerateDemented
                        st.write("The scan shows significant changes consistent with moderate Alzheimer's disease, including noticeable brain tissue loss.")
                    
                    st.caption("Note: This is an AI-based analysis and should not replace professional medical diagnosis.")
                else:
                    st.error("Classifier model not available. Please check the model file.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    # Display information about the model
    with st.expander("About the Classifier Model"):
        st.markdown("""
            This classifier uses a deep learning model based on EfficientNetB0 with transfer learning to detect different 
            stages of Alzheimer's disease from brain MRI scans. The model was trained on the Alzheimer's Dataset with 
            four classes of images.
            
            Key features:
            - Uses transfer learning from a model pre-trained on ImageNet
            - Fine-tuned specifically for brain MRI analysis
            - Includes data augmentation to improve generalization
            - Trained with regularization techniques to prevent overfitting
            
            For research purposes only. Not intended for clinical diagnosis.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application demonstrates the use of GANs for Alzheimer's "
    "disease detection and visualization. The models can be trained "
    "on actual brain scan datasets for real-world applications."
)
