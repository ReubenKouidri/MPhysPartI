class ArrhythmiaDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 reference_file_csv: str,
                 normalize=True,
                 smoothen=True,
                 trim=True,
                 plot=False,
                 wavelet=None,
                 testing=True,
                 leads: Union[List, np.array, int] = 0
                 ):

        self.data_dir = data_dir
        self.references = pd.read_csv(reference_file_csv)
        self.normalize = normalize
        self.trim = trim
        self.smoothen = smoothen
        self.plot = plot
        self.wavelet = wavelet
        self.leads = torch.tensor(leads)
        self.num_leads = self.leads.numel()
        self.testing = testing
        self.mexh_max_width = 32
        self.cmor_max_width = 128
        self.classes = {0: 'Sinus Rhythm', 1: 'AF', 2: 'I-AVB',
                        3: 'LBBB', 4: 'RBBB', 5: 'PAC',
                        6: 'PVC', 7: 'STD', 8: 'STE'}

        if self.references.shape[0] is not self.__len__():
            self.references = self.references.truncate(after=self.__len__() - 1)  # when using an ordered dataset
        self.targets = torch.as_tensor(self.references['First_label'] - 1, dtype=torch.int32)
        self.names = []

        if self.testing:
            for mat_item in os.listdir(self.data_dir):
                if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
                    record_name = mat_item.rstrip('.mat')
                    self.names.append(record_name)

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
        return torch.tensor(smoother.smooth_data[0])

    def _cwt(self, signal, wavelet):
        if wavelet == 'mexh':
            widths = np.arange(1, self.mexh_max_width + 1)
            img, _ = pywt.cwt(signal, widths, wavelet)
            img = resize(img, (128, 128))
            return img

        elif wavelet == 'cmor':
            widths = np.arange(1, self.cmor_max_width + 1)
            img, _ = pywt.cwt(signal, widths, 'cmor1.5-1')
            img = abs(img)
            img = resize(img, (128, 128))
            return img

    def _process(self, ecg_data):
        trim_length = 2000
        step = 2
        ecg_imgs = np.zeros((self.num_leads, 128, 128))
        base = np.zeros((self.num_leads, trim_length // step))
        for i in np.arange(self.num_leads):
            if self.trim:
                base[i] = self._trim_data(ecg_data[i], trim_length, step)
            if self.normalize:
                base[i] = self._normalize(base[i])
            if self.smoothen:
                base[i] = self._smoothen(base[i])
            if self.wavelet is not None:
                ecg_imgs[i] = self._cwt(base[i], self.wavelet)
        return base, ecg_imgs

    def __getitem__(self, item):  # -> Tuple[torch.tensor, torch.tensor]:
        r"""
            - takes in an integer as the index
            - pulls the 'item-th' file from the data directory
            - ECG is cut to 2000 data points (4 seconds) (if True)
            - ECG is smoothed (if True)
            - returns a tuple: (ECG, target)
        """
        file_path = os.path.join(self.data_dir, self.references.iloc[item, 0])
        data = loadmat(f'{file_path}.mat')
        ecg_data = data['ECG']['data'][0][0]  # this is an array: (12, length)
        self.ecg_data = ecg_data
        ecgs, ecg_imgs = self._process(ecg_data)
        #ecg_data = torch.as_tensor(base)
        target = torch.tensor(self.references.iloc[item, 1] - 1)  # picks out the first label only
        ecg_imgs = torch.as_tensor(ecg_imgs, dtype=torch.float32)
        ecgs = torch.as_tensor(ecgs, dtype=torch.float32)

        if self.testing:
            name = self.references.iloc[item, 0]
            return ecg_imgs, name

        return ecg_imgs, target

    def __len__(self):
        r"""
            Length of the dataset
        """
        return len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))])

    def __repr__(self):
        return "Custom dataset containing ECG data and target labels from CPSC 2018 challenge"
      
