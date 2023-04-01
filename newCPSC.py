from scipy.io import loadmat
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tsmoothie import ConvolutionSmoother
from typing import Any
import importlib
import os

wavelets_module_name = "wavelets"
wavelets_module = importlib.import_module(wavelets_module_name)


class CPSCDataset(Dataset):
    AR_classes = {"SR": 0, "AF": 1, "I-AVB": 2, "LBBB": 3, "RBBB": 4, "PAC": 5, "PVC": 6, "STD": 7, "STE": 8}
    length = 4 * 500  # 4s at sampling frequency of 500Hz

    def __init__(
            self,
            data_dir: str | None = "datasets/cpsc_data/test100",
            reference_path: str | None = "datasets/cpsc_data/reference300.csv",
            normalize: bool | None = True,
            smoothen: bool | None = True,
            trim: bool | None = True,
            lead: int | None = 3,
            test: bool | None = False,
            load_in_memory: bool | None = True  # if True, dataset is saved in memory for faster access
    ):
        super(CPSCDataset, self).__init__()
        self.test = test
        self.data_dir = data_dir
        self.references = pd.read_csv(reference_path).fillna(0)[:len(self)]  # in case reference file longer than data
        self.names = self.references['Recording']
        for heading in self.references.columns.values[1:]:
            self.references[heading] = torch.clamp(torch.as_tensor(self.references[heading], dtype=torch.long) - 1, min=0)
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.lead = torch.tensor(lead - 1)  # leads in [1,12] hence -1 indexes correctly
        self.load_in_memory = load_in_memory

        if self.load_in_memory:
            self.data = []
            filenames = sorted(os.listdir(self.data_dir))
            for filename in filenames:
                if filename.endswith(".mat"):
                    filepath = os.path.join(self.data_dir, filename)
                    data = loadmat(filepath)
                    ecg_data = data['ECG']['data'][0][0][self.lead]
                    self.data.append(torch.as_tensor(self._process_data(ecg_data), dtype=torch.float64))

    def _process_data(self, data: np.ndarray) -> np.ndarray:
        data = self._trim_data(data, self.length, step=2) if self.trim else data
        data = self._normalize(data) if self.normalize else data
        data = self._smoothen(data) if self.smoothen else data
        return data

    @staticmethod
    def _normalize(data):
        return (data - min(data)) / (max(data) - min(data))

    @staticmethod
    def _trim_data(data, size, step=1):
        return data[:size:step]

    @staticmethod
    def _smoothen(data):
        smoother = ConvolutionSmoother(window_len=8, window_type='ones')
        smoother.smooth(data)
        return smoother.smooth_data[0]

    def __getitem__(self, item: int) -> tuple[Tensor, Any, Any] | tuple[Tensor, Any]:
        if not self.load_in_memory:
            file_name = os.path.join(self.data_dir, self.references.iloc[item, 0])
            data = loadmat(f'{file_name}.mat')
            ecg_data = data['ECG']['data'][0][0][self.lead]
            ecg_data = torch.as_tensor(self._process_data(ecg_data), dtype=torch.float64)
            return ecg_data, self.references.iloc[item, 1:] if self.test else ecg_data, self.references.iloc[item, 1]
        else:
            return self.data[item], self.references.iloc[item, 1:] if self.test else self.data[item], self.references.iloc[item, 1]

    def __len__(self):
        return len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))])


class CPSCDataset2D(CPSCDataset):
    wavelets = {"mexh": 64, "cmor": 64}  # max widths of wavelet transform

    def __init__(
            self,
            data_dir: str | None = "datasets/cpsc_data/test100",
            reference_path: str | None = "datasets/cpsc_data/reference300.csv",
            wavelet: str | None = "mexh",
            lead: int | None = 3,
            load_in_memory: bool | None = True
    ) -> None:
        super(CPSCDataset2D, self).__init__(
            data_dir=data_dir,
            reference_path=reference_path,
            lead=lead,
            load_in_memory=load_in_memory
        )
        self.wavelet = wavelet if self.wavelets.__contains__(wavelet) else "mexh"
        self.wavelet_fnc = getattr(wavelets_module, self.wavelet)
        #if self.load_in_memory:


    def __getitem__(self, item: int) -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ecg, ref = super().__getitem__(item)
        ecg_img = self.wavelet_fnc(np.array(ecg), self.wavelets[self.wavelet])
        ecg_img = torch.as_tensor(ecg_img).unsqueeze(dim=0)
        return ecg_img, ref
