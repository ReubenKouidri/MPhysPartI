import time
import json
import torch
import collections
from itertools import product
from typing import List
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
from model import Model
from dataset import ArrhythmiaDataset


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lear = lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lear


def get_num_correct(preds, targets):
    return preds.argmax(dim=1).eq(targets).sum().item()


def percentage_error(total_correct, dataset):
    return 100 - 100 * total_correct / len(dataset)


class RunManager:
    def __init__(self):
        self.epoch_id = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_id = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None
        self.df = pd.DataFrame()
        self.saver = ModelSave(verbose=True, path=f'checkpoint_run_{self.run_id}.pt')

    def get_results(self):
        print(pd.DataFrame.from_dict(
            self.run_data,
            orient='columns').sort_values("accuracy", axis=0, ascending=False)
        )

    def begin_run(self, run, network, loader) -> None:
        self.run_start_time = time.time()
        self.run_params = run
        self.run_id += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        #self.tb.add_graph(self.network, images.to(getattr(run, 'device', 'cpu')))
        #self.tb.add_graph(self.network, images)

    def end_run(self) -> None:
        self.tb.close()
        self.epoch_id = 0

    def begin_epoch(self) -> None:
        self.epoch_start_time = time.time()
        self.epoch_id += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self) -> None:
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / 100 #len(self.loader)
        accuracy = self.epoch_num_correct / 100 #len(self.loader)
        self.tb.add_scalar("Loss", loss, self.epoch_id)
        self.tb.add_scalar("Accuracy", accuracy, self.epoch_id)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_id)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_id)

        results = OrderedDict()
        results["run"] = self.run_id
        results["epoch"] = self.epoch_id
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        self.saver(loss, self.network)

        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        self.df = self.df.from_dict(self.run_data, orient='columns')  #pd.DataFrame.from_dict(self.run_data, orient='columns')
        print(self.df.head())

    def track_loss(self, loss, batch) -> None:
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels) -> None:
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels) -> int:
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName) -> None:
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{fileName}.csv')

        #with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
        #    json.dump(self.run_data, f, ensure_ascii=False, indent=4)


class Controls:
    @staticmethod
    def get_hyperparams():
        hyperparams = collections.OrderedDict(
            lr=[0.01],
            batch_size=[10],
            leads=[1],
            num_workers=[1],
            shuffle=[True],
            device=[torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
        )
        return hyperparams


class RunBuilder:
    @staticmethod
    def get_runs(params) -> List:
        Run = collections.namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        #By calling yield, each time a batch is returned to the device from the dataloader.
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class ModelSave:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if self.verbose:
                self.trace_func("No decrease in val loss...")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def run():
    m = RunManager()
    for run in RunBuilder.get_runs(Controls.get_hyperparams()):
        #saver = ModelSave(verbose=True, path=f'checkpoint_{run_id}.pt')
        device = run.device
        print(device)
        model = Model(attention=True).to(device)
        #for parameter in model.parameters():
        #    print(parameter.device)
        train_set_1_dir_path = '/content/TrainingSet1'
        ref_path = '/content/reference.csv'
        dataset = ArrhythmiaDataset(train_set_1_dir_path, ref_path, leads=run.leads, normalize=True, wavelet='mexh')
        train_loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=True, num_workers=run.num_workers)
        train_loader = DeviceDataLoader(train_loader, device)
        optimizer = optim.SGD(params=model.parameters(), lr=run.lr, momentum=0.9, nesterov=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, min_lr=1e-4)

        m.begin_run(run, model, train_loader)
        for epoch in range(5):
            m.begin_epoch()
            #param_groups = optimizer.param_groups
            #for g in param_groups:
            #    print("g_lr ", g['lr'])
            for i, batch in enumerate(train_loader):
                print("batch ", i, "epoch ", epoch)
                images = batch[0].to(device)
                labels = batch[1].to(device)
                preds = model(images)
                loss = F.cross_entropy(preds, labels, reduction='mean')  # Calculate Loss
                optimizer.zero_grad()
                loss.backward()  # Calculate Gradients
                optimizer.step()  # Update Weights
                m.track_loss(loss, batch)
                m.track_num_correct(preds, labels)
            scheduler.step(m.epoch_loss)
            m.end_epoch()
        m.end_run()
    m.get_results()
    m.save('results')

    
if __name__ == 'main':
  run()
