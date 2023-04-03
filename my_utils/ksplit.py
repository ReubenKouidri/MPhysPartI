import random
import warnings
from torch.utils.data import Subset
import collections.abc as abc
from datasets.CPSCDataset import CPSCDataset2D

random.seed(9834275)


def load_2d_dataset(data_path, ref_path) -> CPSCDataset2D:
    return CPSCDataset2D(data_path, ref_path)


def load_datasets(data_path, ref_path):
    return load_2d_dataset(data_path, ref_path)


def split(dataset, ratio: tuple) -> tuple:  # ratio e.g. (0.8, 0.1, 0.1)
    size = len(dataset)
    indices = list(range(size))
    random.shuffle(indices)
    split_sizes = (int(size * ratio[0]), int(size * ratio[1]), int(size * ratio[2]))

    train_indices = indices[0:split_sizes[0]]
    eval_indices = indices[split_sizes[0]:split_sizes[0] + split_sizes[1]]
    test_indices = indices[split_sizes[0] + split_sizes[1]:]

    train_set = Subset(dataset, indices=list(train_indices))
    eval_set = Subset(dataset, indices=list(eval_indices))
    test_set = Subset(dataset, indices=list(test_indices))

    return train_set, eval_set, test_set


def split_dataset(dataset: CPSCDataset2D, k_split: tuple[int, abc.Sequence]) -> tuple[tuple]:
    k = k_split[0]
    if k < 0:
        message = f"k {k} < 0, rounding to 1"
        warnings.warn(message)
        k = 1

    ratio = tuple(k_split[1])  # sequence
    #ratio = [round(x / sum(ratio), 2) for x in ratio]
    if ratio[0] < ratio[1] or ratio[0] < ratio[2]:
        message = f"splits may not be correct: (train, eval, test) = {ratio}"
        warnings.warn(message)

    return tuple(split(dataset, ratio) for _ in range(k_split[0]))
