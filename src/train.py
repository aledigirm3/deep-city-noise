import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


import config
from dataset import UrbanSoundDataModule
from model import CNNClassifier

def train_cross_validation():
    """Performs 10-fold cross-validation."""
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    test_accuracies = []
    
    for fold in range(1, 11):
        print(f"\n{'='*20} FOLD {fold}/10 {'='*20}")

        # 1. DataModule for the current fold
        data_module = UrbanSoundDataModule(current_fold=fold, batch_size=config.BATCH_SIZE)

        data_module.setup(stage='fit')

        # Calculate the dimensions and the number of batches
        train_samples = len(data_module.train_dataset)
        val_samples = len(data_module.val_dataset)
        test_samples = len(data_module.test_dataset)

        # For training, the number of batches is an integer division thanks to 'drop_last=True'
        train_batches = len(data_module.train_dataloader())
        
        # For validation and test, we calculate batches rounding up
        val_batches = len(data_module.val_dataloader())
        test_batches = len(data_module.test_dataloader())

        print(f"--- Dataset Fold {fold} ---")
        print(f"  Training:   {train_samples} campioni in {train_batches} batch")
        print(f"  Validation: {val_samples} campioni in {val_batches} batch")
        print(f"  Test:       {test_samples} campioni in {test_batches} batch")
        print("-" * 28)

        # 2. Model
        model = CNNClassifier(num_classes=config.NUM_CLASSES, lr=config.LEARNING_RATE)

        # 3. Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath=config.CHECKPOINT_DIR,
            filename=f'model_fold_{fold}'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=True,
            mode='min'
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # 4. Trainer
        trainer = pl.Trainer(
            max_epochs=config.EPOCHS,
            accelerator='auto',  # Use GPU/MPS if available
            devices=1,
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            log_every_n_steps=10
        )

        # 5. Training
        trainer.fit(model, datamodule=data_module)
        
        # 6. Test
        # The classification report is automatically printed by the model
        test_results = trainer.test(ckpt_path='best', datamodule=data_module)
        
        # Save the test accuracy for the final average
        if 'test_acc' in test_results[0]:
            test_accuracies.append(test_results[0]['test_acc'])

    # Print the final cross-validation results
    print(f"\n{'='*20} RISULTATI FINALI CROSS-VALIDATION {'='*20}")
    if test_accuracies:
        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies)
        print(f"Accuratezza media sui 10 fold: {mean_acc:.4f} (+/- {std_acc:.4f})")
    else:
        print("Non è stato possibile calcolare l'accuratezza media.")

if __name__ == '__main__':
    # Set matmul precision for Tensor Cores on Ampere+ GPUs
    torch.set_float32_matmul_precision('medium')
    train_cross_validation()
