import os
import torch
import torchaudio
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

import config

class UrbanSoundDataset(Dataset):
    def __init__(self, dataframe, audio_base_path, label_encoder, feature_type, data_aug_transforms=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_base_path = audio_base_path
        self.label_encoder = label_encoder
        self.feature_type = feature_type.lower()
        self.data_aug_transforms = data_aug_transforms

 
        if self.feature_type == 'mel':
            self.feature_transform = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
                ),
                torchaudio.transforms.AmplitudeToDB()
            )
        elif self.feature_type == 'stft':
            # The output of the Spectrogram is in complex magnitude. We use power=2.
            # Convert the complex output into a real matrix containing the energy of the frequencies.
            self.feature_transform = torchaudio.transforms.Spectrogram(
                n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, power=2
            )
        elif self.feature_type == 'log_stft':
            self.feature_transform = torch.nn.Sequential(
                torchaudio.transforms.Spectrogram(
                    n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, power=2
                ),
                torchaudio.transforms.AmplitudeToDB()
            )
        elif self.feature_type == 'mfcc':
            self.feature_transform = torchaudio.transforms.MFCC(
                sample_rate=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
                melkwargs={'n_fft': config.N_FFT, 'n_mels': config.N_MELS, 'hop_length': config.HOP_LENGTH}
            )
        else:
            raise ValueError(f"Tipo di feature '{self.feature_type}' non supportato.")

        # --- Robust handling of dimensions ---
        self.target_samples = int(config.SAMPLE_RATE * config.DURATION)
        dummy_waveform = torch.randn(1, self.target_samples)
        dummy_feature = self.feature_transform(dummy_waveform)
        self.feature_shape = dummy_feature.shape

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_path = os.path.join(self.audio_base_path, f"fold{row['fold']}", row['slice_file_name'])
        label = self.label_encoder.transform([row['class']])[0]

        try:
            waveform, sr = torchaudio.load(file_path)

            if sr != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = torch.mean(waveform, dim=0, keepdim=True)

            if waveform.shape[1] < self.target_samples:
                pad_size = self.target_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_size))
            else:
                waveform = waveform[:, :self.target_samples]
            
            # 1. Extract features (mel, mfcc, etc.)
            features = self.feature_transform(waveform)

            # 2. Temporal padding/truncation to standardize the width
            if features.shape[-1] < config.EXPECTED_FRAMES:
                pad_size = config.EXPECTED_FRAMES - features.shape[-1]
                features = F.pad(features, (0, pad_size))
            else:
                features = features[:, :, :config.EXPECTED_FRAMES]

            # 3. Apply data augmentation (if present)
            if self.data_aug_transforms:
                features = self.data_aug_transforms(features)

            return features, label

        except Exception as e:
            print(f"Errore nel caricare il file {file_path}: {e}")
            # Returns a zero tensor with the correct shape for the chosen feature type
            return torch.zeros(self.feature_shape), 0


class UrbanSoundDataModule(pl.LightningDataModule):

    def __init__(self, current_fold: int, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.current_fold = current_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_type = config.FEATURE_TYPE
        self.metadata_df = pd.read_csv(config.METADATA_PATH)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.metadata_df['class'].unique())

        self.train_transforms = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
        )
        self.val_test_transforms = None

    def setup(self, stage: str = None):

        train_val_df = self.metadata_df[self.metadata_df['fold'] != self.current_fold]
        self.test_df = self.metadata_df[self.metadata_df['fold'] == self.current_fold]
        
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=config.VALIDATION_SPLIT_RATIO, 
            stratify=train_val_df['class'], random_state=config.RANDOM_STATE
        )

        # Pass the 'feature_type' when creating the datasets
        common_args = {
            "audio_base_path": config.AUDIO_BASE_PATH,
            "label_encoder": self.label_encoder,
            "feature_type": self.feature_type
        }

        self.train_dataset = UrbanSoundDataset(self.train_df, data_aug_transforms=self.train_transforms, **common_args)
        self.val_dataset = UrbanSoundDataset(self.val_df, data_aug_transforms=None, **common_args)
        self.test_dataset = UrbanSoundDataset(self.test_df, data_aug_transforms=None, **common_args)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)