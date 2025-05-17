import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, applications
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Configure TensorFlow
tf.compat.v1.disable_eager_execution()  # Disable eager execution for better compatibility
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TF v1 logging
tf.get_logger().setLevel('ERROR')  # Suppress TF v2 logging

# Custom JSON encoder for TensorFlow tensors
class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        if hasattr(obj, 'numpy'):
            return obj.numpy().tolist()
        return super(TensorEncoder, self).default(obj)

# Monkey patch TensorFlow's JSON serialization to handle tensors
_original_json_dumps = json.dumps
def _patched_json_dumps(obj, *args, **kwargs):
    return _original_json_dumps(obj, *args, **kwargs, cls=TensorEncoder)

# Apply the patch
json.dumps = _patched_json_dumps

class AlzheimerClassifier:
    def __init__(self, img_shape=(128, 128, 3), num_classes=4):
        """
        Initialize multi-class Alzheimer's classifier.
        
        Args:
            img_shape: Shape of the input images (height, width, channels)
            num_classes: Number of classes (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
        """
        self.img_shape = img_shape
        self.num_classes = num_classes
        
        # Build the model
        self.model = self.build_model()
        
        # Compile the model with TensorFlow v1 compatibility
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
    
    def build_model(self):
        """
        Build a CNN classifier model using transfer learning with a pre-trained EfficientNetV2B0
        and attention mechanisms.
        
        Returns:
            Compiled Keras model for Alzheimer's classification
        """
        # Use pre-trained EfficientNetV2B0 as the base model
        base_model = applications.EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Input layer
        inputs = layers.Input(shape=self.img_shape)
        
        # Base model
        x = base_model(inputs)
        
        # Attention mechanism
        attention = layers.Conv2D(1, kernel_size=1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        # Global pooling with residual connection
        pooled = layers.GlobalAveragePooling2D()(x)
        residual = layers.Dense(1280)(pooled)  # EfficientNetV2B0 output size
        
        # Dense layers with residual connections
        x = layers.Dense(1024, activation='relu')(pooled)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Residual connection 1
        x = layers.Add()([x, layers.Dense(1024)(residual)])
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Residual connection 2
        x = layers.Add()([x, layers.Dense(512)(residual)])
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Final classification layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def fine_tune(self, unfreeze_layers=150):
        """
        Fine-tune the model by unfreezing some of the top layers of the base model.
        
        Args:
            unfreeze_layers: Number of top layers to unfreeze for fine-tuning
        """
        # Unfreeze the specified number of layers from the end of the base model
        base_model = self.model.layers[1]
        for layer in base_model.layers[-(unfreeze_layers):]:
            layer.trainable = True
        
        # Recompile the model with TensorFlow v1 compatibility
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=5e-7),
            metrics=['accuracy']
        )
    
    def train(self, train_data, train_labels, validation_data=None, 
              validation_labels=None, epochs=20, batch_size=32, 
              callbacks=None, fine_tune_epochs=10):
        """
        Train the Alzheimer's classifier.
        
        Args:
            train_data: Training images
            train_labels: Training labels (one-hot encoded)
            validation_data: Validation images
            validation_labels: Validation labels (one-hot encoded)
            epochs: Number of initial training epochs
            batch_size: Batch size
            callbacks: Optional list of Keras callbacks
            fine_tune_epochs: Number of fine-tuning epochs after initial training
            
        Returns:
            Training history
        """
        # Create default callbacks if none provided
        if callbacks is None:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    'best_alzheimer_classifier', 
                    save_best_only=True, 
                    monitor='val_accuracy',
                    save_weights_only=True,
                ),
                keras.callbacks.EarlyStopping(
                    patience=5, 
                    monitor='val_loss', 
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.2, 
                    patience=3, 
                    monitor='val_loss', 
                    min_lr=1e-6
                )
            ]
        
        # Initial training phase
        print("Initial training phase...")
        history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(validation_data, validation_labels) if validation_data is not None else None,
            callbacks=callbacks
        )
        
        # Convert tensors to Python types for initial history immediately after creation
        for key in history.history:
            history.history[key] = [float(val.numpy()) if hasattr(val, 'numpy') else float(val) 
                                   for val in history.history[key]]
        
        # Fine-tuning phase
        if fine_tune_epochs > 0:
            print("\nFine-tuning phase...")
            self.fine_tune()
            
            fine_tune_history = self.model.fit(
                train_data, train_labels,
                batch_size=batch_size,
                epochs=fine_tune_epochs,
                validation_data=(validation_data, validation_labels) if validation_data is not None else None,
                callbacks=callbacks
            )
            
            # Convert tensors to Python types for fine-tune history before extending
            for key in fine_tune_history.history:
                converted_values = [float(val.numpy()) if hasattr(val, 'numpy') else float(val) 
                                  for val in fine_tune_history.history[key]]
                history.history[key].extend(converted_values)
        
        return history
    
    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test images
            test_labels: Test labels (one-hot encoded)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get model predictions
        predictions = self.model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(test_labels, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get classification report
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(test_data, test_labels, verbose=0)
        
        # Convert any tensors to standard Python types
        if hasattr(test_loss, 'numpy'):
            test_loss = float(test_loss.numpy())
        if hasattr(test_acc, 'numpy'):
            test_acc = float(test_acc.numpy())
            
        # Ensure predictions are converted from tensors if needed
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history.
        
        Args:
            history: Training history from model.fit()
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_path: Optional path to save the plot
        """
        if class_names is None:
            class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        plt.figure(figsize=(10, 8))
        try:
            # Try using seaborn for a nicer visualization
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=class_names, yticklabels=class_names)
        except ImportError:
            # Fallback to matplotlib
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Add text annotations
            thresh = cm.max() / 2
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
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def predict_class(self, image):
        """
        Predict the class of a single image.
        
        Args:
            image: Input image (should be preprocessed to match model input shape)
            
        Returns:
            Dictionary with predicted class index, class name, and probabilities
        """
        # Make sure image has the right shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get prediction
        pred = self.model.predict(image)
        
        # Convert tensor to numpy if needed
        if hasattr(pred, 'numpy'):
            pred = pred.numpy()
            
        pred_class = np.argmax(pred, axis=1)[0]
        
        # Map class index to name
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        pred_class_name = class_names[pred_class]
        
        return {
            'class_index': int(pred_class),  # Ensure this is a plain Python int
            'class_name': pred_class_name,
            'probabilities': pred[0].tolist()  # Convert numpy array to list for JSON serialization
        }
    
    def save_model(self, save_path='alzheimer_classifier_model'):
        """
        Save the model to a file.
        
        Args:
            save_path: Path to save the model
        """
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save weights first to avoid serialization issues
        weights_path = save_path + '_weights'
        self.model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")
        
        # Try to save full model architecture as JSON
        try:
            # Save model architecture as JSON
            with open(save_path + '_architecture.json', 'w') as f:
                model_json = self.model.to_json()
                f.write(model_json)
            print(f"Model architecture saved to {save_path}_architecture.json")
        except Exception as e:
            print(f"Warning: Could not save model architecture: {e}")
    
    def load_model(self, model_path='alzheimer_classifier_model'):
        """
        Load the model from a file.
        
        Args:
            model_path: Path to the saved model
        """
        # Try to load model weights
        weights_path = model_path + '_weights'
        if os.path.exists(weights_path) or os.path.exists(weights_path + '.index'):
            # We already have a model instance, just load weights
            self.model.load_weights(weights_path)
            print(f"Model weights loaded from {weights_path}")
        else:
            # Try loading as a whole model
            try:
                self.model = models.load_model(model_path)
                print(f"Full model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
    @staticmethod
    def preprocess_image(image, target_size=(128, 128)):
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            target_size: Target size for the image
            
        Returns:
            Preprocessed image ready for model input
        """
        # Load image if it's a file path
        if isinstance(image, str):
            from keras.utils import load_img, img_to_array
            image = load_img(image, target_size=target_size)
            image = img_to_array(image)
        
        # Resize if needed
        if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
            import cv2
            image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply EfficientNet preprocessing
        image = applications.efficientnet.preprocess_input(image)
        
        return image 