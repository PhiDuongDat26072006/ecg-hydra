import math
import random

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.data.components.ecg_dataset import ECGDataset


def Tnoise_powerline(fs=100, N=1000, C=1, fn=50.0, K=3, channels=1):
    t = torch.arange(0, N / fs, 1.0 / fs)

    signal = torch.zeros(N)
    phi1 = random.uniform(0, 2 * math.pi)
    for k in range(1, K + 1):
        ak = random.uniform(0, 1)
        signal += C * ak * torch.cos(2 * math.pi * k * fn * t + phi1)
    signal = C * signal[:, None]
    if channels > 1:
        channel_gains = torch.empty(channels).uniform_(-1, 1)
        signal = signal * channel_gains[None]
    return signal


def Tnoise_baseline_wander(
    fs=100,
    N=1000,
    C=1.0,
    fc=0.5,
    fdelta=0.01,
    channels=1,
    independent_channels=False,
):
    if fdelta is None:
        fdelta = fs / N

    K = int((fc / fdelta) + 0.5)
    t = torch.arange(0, N / fs, 1.0 / fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2 * math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2 * math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C * res


class ChannelResize:
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range))
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T
        wave = resize_factors * wave
        return wave


class GaussianNoise:
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob

    def __call__(self, wave):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave


class BaselineShift:
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave


class BaselineWander:
    def __init__(self, prob=1.0, freq=1000, C=1.0):
        self.freq = freq
        self.prob = prob
        self.C = C

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave, C=self.C)
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave


class PowerlineNoise:
    def __init__(self, prob=1.0, freq=1000, C=1.0):
        self.freq = freq
        self.prob = prob
        self.C = C

    def __call__(self, wave):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(fs=self.freq, N=len_wave, C=self.C, channels=channels).T
            wave = wave + noise
        return wave


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave):
        for t in self.transforms:
            wave = t(wave)
        return wave


class ECGDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_file: str,
        data_dir: str,
        fold_train,
        fold_test,
        batch_size: int,
        num_workers: int,
        split_ratio: float = 0.9,
        sample_before: int = 100,
        sample_after: int = 100,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir

        if isinstance(fold_test, int):
            self.fold_test = [fold_test]
        else:
            self.fold_test = fold_test

        if fold_train is None:
            self.fold_train = [i for i in range(5) if i not in self.fold_test]
        else:
            self.fold_train = fold_train

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.sample_before = sample_before
        self.sample_after = sample_after
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_train,
            sample_before=self.sample_before,
            sample_after=self.sample_after,
            transform=ComposeTransforms(
                [
                    BaselineWander(prob=0.5, C=0.0001),
                    GaussianNoise(prob=0.5, scale=0.0001),
                    PowerlineNoise(prob=0.5, C=0.0001),
                    ChannelResize(magnitude_range=(0.5, 2.0)),
                    BaselineShift(prob=0.5, scale=0.01),
                ]
            ),
        )

        split_with_patient = False
        if split_with_patient:
            dataset_patient_id_list = list(dataset.info["patient_number"].unique())
            random.shuffle(dataset_patient_id_list)
            split_point = int(len(dataset_patient_id_list) * self.split_ratio)
            train_patient_id_list = dataset_patient_id_list[:split_point]
            train_indices = dataset.info[dataset.info["patient_number"].isin(train_patient_id_list)].index.tolist()
            val_indices = dataset.info[~dataset.info["patient_number"].isin(train_patient_id_list)].index.tolist()
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        else:
            total_size = len(dataset)
            train_size = int(total_size * self.split_ratio)
            val_size = total_size - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        self.test_dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_test,
            sample_before=self.sample_before,
            sample_after=self.sample_after,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

