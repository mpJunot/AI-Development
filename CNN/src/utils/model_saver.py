import os
import pickle
from typing import Dict, Any

class ModelSaver:
    def __init__(self, save_dir: str = "saved_models"):
        """
        Initialize the model saver.
        :param save_dir: Directory where models will be saved
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_model(self, model: Any, filename: str, metadata: Dict = None) -> None:
        """
        Save the model and its parameters to disk.
        :param model: The model to save
        :param filename: Name of the file to save the model
        :param metadata: Additional metadata to save with the model
        """
        filepath = os.path.join(self.save_dir, filename)

        model_state = {
            'layers': [],
            'metadata': {
                'input_shape': getattr(model, 'input_shape', None),
                'num_classes': getattr(model, 'num_classes', None)
            }
        }

        if metadata:
            model_state['metadata'].update(metadata)

        if model_state['metadata']['input_shape'] is None or model_state['metadata']['num_classes'] is None:
            raise ValueError("Model must have input_shape and num_classes attributes")

        for layer in model.layers:
            layer_state = {
                'type': layer.__class__.__name__,
                'params': {}
            }

            for attr in ['weights', 'biases', 'gamma', 'beta', 'running_mean', 'running_var',
                        'momentum', 'epsilon', 'p', 'alpha', 'in_channels', 'out_channels',
                        'kernel_size', 'stride', 'padding', 'pool_size', 'input_dim',
                        'output_dim', 'num_features']:
                if hasattr(layer, attr):
                    layer_state['params'][attr] = getattr(layer, attr)

            model_state['layers'].append(layer_state)

        # Save the model state using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

    def load_model(self, filename: str, model_class: Any) -> Any:
        """
        Load a model from disk.
        :param filename: Name of the file to load
        :param model_class: The class of the model to instantiate
        :return: The loaded model
        """
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load the model state
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # Get metadata
        metadata = model_state.get('metadata', {})
        input_shape = metadata.get('input_shape')
        num_classes = metadata.get('num_classes')

        if input_shape is None or num_classes is None:
            raise ValueError(f"Model metadata missing required fields. Found: {metadata}")

        model = model_class(
            input_shape=input_shape,
            num_classes=num_classes
        )

        for layer, layer_state in zip(model.layers, model_state['layers']):
            params = layer_state['params']
            for key, value in params.items():
                if hasattr(layer, key):
                    setattr(layer, key, value)

        return model

    def save_checkpoint(self, model: Any, optimizer: Any, epoch: int, loss: float, filename: str) -> None:
        """
        Save a training checkpoint.
        :param model: The model to save
        :param optimizer: The optimizer state
        :param epoch: Current epoch number
        :param loss: Current loss value
        :param filename: Name of the checkpoint file
        """
        checkpoint = {
            'model_state': self.save_model(model, filename),
            'optimizer_state': optimizer.__dict__,
            'epoch': epoch,
            'loss': loss
        }

        filepath = os.path.join(self.save_dir, f"checkpoint_{filename}")
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filename: str, model_class: Any, optimizer_class: Any) -> tuple:
        """
        Load a training checkpoint.
        :param filename: Name of the checkpoint file
        :param model_class: The class of the model to instantiate
        :param optimizer_class: The class of the optimizer to instantiate
        :return: Tuple of (model, optimizer, epoch, loss)
        """
        filepath = os.path.join(self.save_dir, f"checkpoint_{filename}")

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        model = self.load_model(filename, model_class)
        optimizer = optimizer_class(**checkpoint['optimizer_state'])

        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
