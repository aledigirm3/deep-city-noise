import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import numpy as np

import config

class CNNClassifier(pl.LightningModule):
    def __init__(self, num_classes=config.NUM_CLASSES, lr=config.LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        
        # Test outputs for the final classification report
        self.test_step_outputs = []

        self.conv_block1 = self._create_conv_block(1, 32)
        self.conv_block2 = self._create_conv_block(32, 64)
        self.conv_block3 = self._create_conv_block(64, 128)
        
        # Adaptive pooling to make the model independent of the exact input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), # The input size is now always 128 thanks to Adaptive Pooling
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc, logits.argmax(dim=1), y

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, _, _ = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc, preds, targets = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.test_step_outputs.append({'preds': preds, 'targets': targets})
        return {'preds': preds, 'targets': targets}

    def on_test_epoch_end(self):
        # Concatenate the results from all test batches
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).cpu().numpy()
        
        # Get class names from the datamodule
        target_names = self.trainer.datamodule.label_encoder.classes_
        
        # Print the report
        print("\n--- Classification Report ---")
        print(classification_report(all_targets, all_preds, target_names=target_names))
        
        # Clears outputs for the next test run (e.g., in cross-validation)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=config.LR_SCHEDULER_PATIENCE, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}