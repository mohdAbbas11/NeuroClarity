import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.metrics import BinaryAccuracy, AUC, Precision, Recall
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Try to import seaborn, but provide a fallback if it's not available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Seaborn not installed. Using matplotlib for visualizations instead.")

def evaluate_classifier(model, test_data, test_labels, threshold=0.5):
    """
    Evaluate a classifier model using various metrics.
    
    Args:
        model: Trained classifier model
        test_data: Test data
        test_labels: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    predictions = model.predict(test_data)
    
    # Compute metrics
    binary_acc = BinaryAccuracy(threshold=threshold)
    binary_acc.update_state(test_labels, predictions)
    
    auc_metric = AUC()
    auc_metric.update_state(test_labels, predictions)
    
    precision_metric = Precision(thresholds=threshold)
    precision_metric.update_state(test_labels, predictions)
    
    recall_metric = Recall(thresholds=threshold)
    recall_metric.update_state(test_labels, predictions)
    
    # Convert predictions to binary using threshold
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, binary_preds)
    
    # Calculate F1 score
    f1 = 2 * (precision_metric.result().numpy() * recall_metric.result().numpy()) / \
        (precision_metric.result().numpy() + recall_metric.result().numpy() + 1e-7)
    
    return {
        'accuracy': binary_acc.result().numpy(),
        'auc': auc_metric.result().numpy(),
        'precision': precision_metric.result().numpy(),
        'recall': recall_metric.result().numpy(),
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for key in history.history.keys():
        if 'loss' in key:
            plt.plot(history.history[key], label=key)
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    for key in history.history.keys():
        if 'loss' not in key:
            plt.plot(history.history[key], label=key)
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(cm, class_names=['Normal', 'Alzheimer'], save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        # Fallback to matplotlib if seaborn is not available
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def evaluate_image_quality(original_images, generated_images):
    """
    Evaluate the quality of generated images using PSNR and SSIM.
    
    Args:
        original_images: Original images
        generated_images: Generated images
        
    Returns:
        Dictionary containing PSNR and SSIM values
    """
    # Make sure the images are in the range [0, 1]
    original_images = (original_images + 1) / 2.0
    generated_images = (generated_images + 1) / 2.0
    
    psnr_values = []
    ssim_values = []
    
    for i in range(len(original_images)):
        psnr = tf.image.psnr(original_images[i], generated_images[i], max_val=1.0).numpy()
        ssim = tf.image.ssim(original_images[i], generated_images[i], max_val=1.0).numpy()
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    return {
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values)
    } 