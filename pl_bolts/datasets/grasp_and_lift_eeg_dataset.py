import os
import torch
import torch.utils.data as data


class GraspAndLiftEEGDataset(data.Dataset):
    """ 32-channel, 500Hz EEG dataset of subjects performing a various
    grasp-and-lift motor tasks, with per-sample class labels.

    There are 12 subjects in total, 10 series of trials for each subject,
    and approximately 30 trials within each series. The number of trials
    varies for each series. The training set contains the first 8 series
    for each subject. The test set contains the 9th and 10th series.

    Videos of trials being performed:
    https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2014.47/MediaObjects/41597_2014_BFsdata201447_MOESM69_ESM.avi
    https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2014.47/MediaObjects/41597_2014_BFsdata201447_MOESM70_ESM.avi
    (Note: these videos are also available in the S3 bucket)

    Args:
        root: Path to directory containing train/ and test/ folders

        train: If true, use the train/ directory and load class labels.
            If false, load the test/ directory (which lacks labels)

        download: If true, download the data if it is not present locally

        num_samples: The number of samples in the returned examples.
            If None, the the dataset yields the full length trials.

        last_label_only: If true, return only the last sample's labels.
            Must be used with num_samples.

    Examples:
        >>> dataset = GraspAndLiftEEGDataset('/data', train=True, download=True, num_samples=1024)
        Downloading from https://grasplifteeg.nyc3.digitaloceanspaces.com/grasp-and-lift-eeg-detection.zip
        Downloaded in 283 seconds
        Extracting /data/grasp-and-lift-eeg-detection.zip to /data
        Unzipped in 36 seconds
        >>> len(dataset)
        17887546
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([32, 1024])
        >>> label.shape
        torch.Size([6, 1024])

    Labels:
        1. HandStart
        2. FirstDigitTouch
        3. BothStartLoadPhase
        4. LiftOff
        5. Replace
        6. BothReleased

    Reference:
        Luciw, M., Jarocka, E. & Edin, B. Multi-channel EEG recordings during 3,936
            grasp and lift trials with varying weight and friction. Sci Data 1, 140047
            (2014). https://doi.org/10.1038/sdata.2014.47

        https://www.kaggle.com/c/grasp-and-lift-eeg-detection

    License:
        Original data has been made available under the terms of Attribution 4.0
        International Creative Commons License (http://creativecommons.org/licenses/by/4.0/).
    """

    GRASPLIFT_DATA_HEADER = 'id,Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,TP9,CP5,CP1,CP2,CP6,' + \
        'TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10\n'

    GRASPLIFT_EVENTS_HEADER = 'id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n'

    ZIP_URL = 'https://grasplifteeg.nyc3.digitaloceanspaces.com/grasp-and-lift-eeg-detection.zip'

    ZIP_SIZE_BYTES = 980887394

    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = True,
                 num_samples: int = None,
                 last_label_only: bool = False):
        super(GraspAndLiftEEGDataset, self).__init__()
        if num_samples is None and last_label_only:
            raise ValueError('last_label_only cannot be used without setting num_samples')
        self.num_samples = num_samples
        self.last_label_only = last_label_only
        dir = os.path.join(root, 'train' if train else 'test')
        if not os.path.exists(dir):
            if not download:
                raise ValueError(f'{dir} does not exist')
            if not os.path.exists(root):
                os.makedirs(root)
            self.download(root)
        csv_suffix = '.csv'
        bin_suffix = '.csv.bin'
        csv_files = [os.path.join(dp, f)
                     for dp, dn, fn in os.walk(os.path.expanduser(dir))
                     for f in fn
                     if f.endswith(csv_suffix)]
        bin_files = [os.path.join(dp, f)
                     for dp, dn, fn in os.walk(os.path.expanduser(dir))
                     for f in fn
                     if f.endswith(bin_suffix)]

        should_compile = False

        if len(bin_files) < len(csv_files):
            print(f'Number of .csv.bin files ({len(bin_files)}) '
                  f'is less than the number of .csv ({len(csv_files)}).'
                  ' Compiling binary representation...')
            should_compile = True

        if should_compile:
            self.X, self.Y = self.compile_bin(csv_files)
            if num_samples is not None:
                # Divide each example up into windows
                self.total_examples = 0
                for x in self.X:
                    self.total_examples += x.shape[1] - num_samples + 1
        else:
            examples = {}
            self.total_examples = 0
            for file in bin_files:
                is_data = file.endswith('_data.csv.bin')
                series = file[:-len('_data.csv.bin')
                              if is_data else -len('_events.csv.bin')]
                samples = torch.load(file)
                item = examples.get(series, [None, None])
                item[0 if is_data else 1] = samples
                examples[series] = item
                if is_data and num_samples is not None:
                    self.total_examples += samples.shape[1] - num_samples + 1
            self.X = []
            Y = []
            for series in sorted(examples):
                x, y = examples[series]
                self.X.append(x)
                if y is not None:
                    Y.append(y)
            self.Y = Y if len(Y) > 0 else None

    def download(self, root: str):
        import requests
        import time
        import zipfile
        zip_path = os.path.join(root, 'grasp-and-lift-eeg-detection.zip')
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) != self.ZIP_SIZE_BYTES:
            print(f'Downloading from {self.ZIP_URL}')
            start = time.time()
            r = requests.get(self.ZIP_URL)
            if r.status_code != 200:
                raise ValueError(
                    f'Expected status code 200, got {r.status_code}')
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            delta = time.time() - start
            print(f'Downloaded in {int(delta)} seconds')
        print(f'Extracting {zip_path} to {root}')
        start = time.time()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        delta = time.time() - start
        print(f'Unzipped in {int(delta)} seconds')
        os.remove(zip_path)

    def compile_bin(self, csv_files: list):
        examples = {}
        for i, file in enumerate(csv_files):
            is_data = file.endswith('_data.csv')
            samples = []
            with open(file, 'r') as f:
                hdr = f.readline()
                expected_hdr = self.GRASPLIFT_DATA_HEADER if is_data else self.GRASPLIFT_EVENTS_HEADER
                if hdr != expected_hdr:
                    raise ValueError('bad header')
                for line in f:
                    channels = line.strip().split(',')[1:]
                    if is_data:
                        # Data is converted to float eventually anyway
                        channels = [float(x) for x in channels]
                    else:
                        # Labels are integer format
                        channels = [int(x) for x in channels]
                    channels = torch.Tensor(channels).unsqueeze(1)
                    samples.append(channels)
            samples = torch.cat(samples, dim=1)
            series = file[:-len('_data.csv')
                          if is_data else -len('_events.csv')]
            item = examples.get(series, [None, None])
            item[0 if is_data else 1] = samples
            examples[series] = item
            print(f'Processed {i+1}/{len(csv_files)} {file}')
        X = []
        Y = []
        for series in sorted(examples):
            x, y = examples[series]
            torch.save(samples, series + '_data.csv.bin')
            X.append(x)
            if y is not None:
                torch.save(samples, series + '_events.csv.bin')
                Y.append(y)
        return X, Y if len(Y) > 0 else None

    def __getitem__(self, index):
        if self.num_samples is None:
            # Return the entire example (e.g. reinforcement learning)
            return (self.X[index], self.Y[index] if self.Y is not None else [])
        # Find the example and offset for the index
        ofs = 0
        for i, x in enumerate(self.X):
            num_examples = x.shape[1] - self.num_samples + 1
            if index >= ofs + num_examples:
                ofs += num_examples
                continue
            j = index - ofs
            x = x[:, j:j + self.num_samples]
            if self.Y is not None:
                if self.last_label_only:
                    # Only return the label for the last sample
                    y = self.Y[i][:, j + self.num_samples - 1]
                else:
                    # Return labels for all samples
                    y = self.Y[i][:, j:j + self.num_samples]
            else:
                y = []
            return x, y
        raise ValueError(f'unable to seek {index}')

    def __len__(self):
        if self.num_samples is None:
            # No windowing - each example is full length
            return len(self.X)
        # Use precalculated dataset length
        return self.total_examples
