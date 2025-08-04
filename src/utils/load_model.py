import torch
import os
from models.architectures.snn import SpikingNeuralNetwork
from models.architectures.lstm import LSTMClassifier
from models.architectures.cnn import CNNClassifier


def load_model_checkpoint(checkpoint_path, device='cpu', load_optimizer=False):
    """
    Load a model checkpoint and return the model (and optionally optimizer)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['model_config']
    model_name = checkpoint['model_name']

    if 'SNN' in model_name and 'SRNN' not in model_name:
        num_layers = int(model_name.split('_')[1][0])
        model = SpikingNeuralNetwork(
            input_size=config['input_size'] or 700,
            hidden_size=config['hidden_size'] or 128,
            output_size=config['output_size'] or 10,
            num_layers=num_layers,
            recurrent=False,
            dt=config['dt'] or 1e-3
        )
    elif model_name == 'SRNN':
        model = SpikingNeuralNetwork(
            input_size=config['input_size'] or 700,
            hidden_size=config['hidden_size'] or 128,
            output_size=config['output_size'] or 10,
            recurrent=True,
            dt=config['dt'] or 1e-3
        )
    elif model_name == 'LSTM':
        model = LSTMClassifier(
            input_size=config['input_size'] or 700,
            hidden_size=config['hidden_size'] or 128,
            output_size=config['output_size'] or 10,
            dropout=0.2
        )
    elif model_name == 'CNN':
        model = CNNClassifier(
            input_channels=64,
            output_size=config['output_size'] or 10,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✅ Model {model_name} loaded from {checkpoint_path}")
    print(
        f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}, Accuracy: {checkpoint['accuracy']:.4f}")
    print(f"   Saved: {checkpoint['timestamp']}")

    if load_optimizer:
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint

    return model, checkpoint


def load_all_models_from_checkpoints(checkpoint_dir='checkpoints', device='cpu'):
    """
    Load all models from their best checkpoints
    """
    models = {}
    checkpoints = {}

    model_names = ['SNN_1Layer', 'SNN_2Layer',
                   'SNN_3Layer', 'SRNN', 'LSTM', 'CNN']

    for model_name in model_names:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'{model_name}_best.pth')

        if os.path.exists(checkpoint_path):
            try:
                model, checkpoint = load_model_checkpoint(
                    checkpoint_path=checkpoint_path,
                    device=device
                )
                models[model_name] = model
                checkpoints[model_name] = checkpoint
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")

    return models, checkpoints


def load_specific_model(model_name, checkpoint_dir='checkpoints', device='cpu', epoch=None):
    """
    Load a specific model, optionally from a specific epoch
    """
    if epoch is not None:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
    else:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'{model_name}_best.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint = load_model_checkpoint(
        model_class=None,
        checkpoint_path=checkpoint_path,
        device=device
    )

    return model, checkpoint
