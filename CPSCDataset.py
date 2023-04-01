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
            data_path: str | None = "datasets/cpsc_data/test100",
            reference_path: str | None = "datasets/cpsc_data/reference300.csv",
            normalize: bool | None = True,
            smoothen: bool | None = True,
            trim: bool | None = True,
            lead: int | None = 3,
            test: bool | None = False
    ):
        super(CPSCDataset, self).__init__()
        self.test = test
        self.data_path = data_path
        self.references = pd.read_csv(reference_path).fillna(0)
        self.names = self.references['Recording']
        for heading in self.references.columns.values[1:]:
            self.references[heading] = torch.clamp(torch.as_tensor(self.references[heading], dtype=torch.long) - 1, min=0)
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.lead = torch.tensor(lead - 1)  # leads in [1,12] hence -1 indexes correctly

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
        file_name = os.path.join(self.data_path, self.references.iloc[item, 0])
        data = loadmat(f'{file_name}.mat')
        ecg_data = data['ECG']['data'][0][0][self.lead]

        base = self._trim_data(ecg_data, self.length, step=2) if self.trim else ecg_data
        base = self._normalize(base) if self.normalize else base
        base = self._smoothen(base) if self.smoothen else base
        ecg_data = torch.as_tensor(base, dtype=torch.float64)

        if self.test:
            return ecg_data, self.references.iloc[item, 1:]
        return ecg_data, self.references.iloc[item, 1]

    def __len__(self):
        return len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])


class CPSCDataset2D(CPSCDataset):
    wavelets = {"mexh": 64, "cmor": 64}  # max widths of wavelet transform

    def __init__(
            self,
            data_path: str | None = "datasets/cpsc_data/test100",
            reference_path: str | None = "datasets/cpsc_data/reference300.csv",
            wavelet: str | None = "mexh",
            lead: int | None = 3
    ) -> None:
        super(CPSCDataset2D, self).__init__(data_path, reference_path, lead)
        self.wavelet = wavelet if self.wavelets.__contains__(wavelet) else "mexh"
        self.wavelet_fnc = getattr(wavelets_module, self.wavelet)

    def __getitem__(self, item: int) -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ecg, ref = super().__getitem__(item)
        ecg_img = self.wavelet_fnc(np.array(ecg), self.wavelets[self.wavelet])
        ecg_img = torch.as_tensor(ecg_img).unsqueeze(dim=0)
        return ecg_img, ref
