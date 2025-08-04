import torch
import os
from datetime import datetime


def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, model_name, checkpoint_dir='../models/checkpoints'):
    """
    Save a comprehensive checkpoint for any model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'input_size': getattr(model, 'input_size', None),
            'hidden_size': getattr(model, 'hidden_size', None),
            'output_size': getattr(model, 'output_size', None),
            'num_layers': getattr(model, 'num_layers', None),
            'recurrent': getattr(model, 'recurrent', None),
            'dt': getattr(model, 'dt', None),
        }
    }

    checkpoint_path = os.path.join(
        checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
    torch.save(checkpoint, best_path)

    print(f"✅ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def save_all_models_checkpoints(models, optimizers, epoch, losses, accuracies, checkpoint_dir='../models/checkpoints'):
    """
    Save checkpoints for all models at once
    """
    saved_paths = {}

    for model_name in models.keys():
        path = save_model_checkpoint(
            model=models[model_name],
            optimizer=optimizers[model_name],
            epoch=epoch,
            loss=losses[model_name],
            accuracy=accuracies[model_name],
            model_name=model_name,
            checkpoint_dir=checkpoint_dir
        )
        saved_paths[model_name] = path

    return saved_paths
