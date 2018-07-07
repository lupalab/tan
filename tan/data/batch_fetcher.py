import numpy as np
from ..utils import misc


class BatchFetcher:

    def __init__(self, *datasets, **kwargs):
        self._datasets = datasets
        self.ndatasets = len(datasets)
        if len(datasets) == 1:
            self._shape = datasets[0].shape
            self.dim = self._shape[-1]
        else:
            self._shape = [d.shape for d in datasets]
            self.dim = [s[-1] for s in self._shape]
        self._N = datasets[0].shape[0]
        self._perm = np.random.permutation(self._N)
        self._curri = 0
        self._loop_around = misc.get_default(kwargs, 'loop_around', True)

    def reset_index(self):
        self._curri = 0

    def next_batch(self, batch_size):
        assert self._N > batch_size

        curri = self._curri
        if self._loop_around:
            endi = (curri+batch_size) % self._N
        else:
            if curri >= self._N:
                raise IndexError
            endi = np.minimum(curri+batch_size, self._N)
        if endi < curri:  # looped around
            inds = np.concatenate((np.arange(curri, self._N), np.arange(endi)))
        else:
            inds = np.arange(curri, endi)
        self._curri = endi

        if self._loop_around:
            batches = tuple(d[self._perm[inds]] for d in self._datasets)
        else:
            batches = tuple(d[inds] for d in self._datasets)
        if len(batches) == 1:
            return batches[0]
        return batches


class DatasetFetchers:

    def __init__(self, train, validation, test):
        self.train = BatchFetcher(*train)
        self.validation = BatchFetcher(*validation)
        self.test = BatchFetcher(*test, loop_around=False)

    def reset_index(self):
        self.train.reset_index()
        self.validation.reset_index()
        self.test.reset_index()

    @property
    def dim(self):
        return self.train.dim
