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
        self._subsamp = misc.get_default(kwargs, 'subsamp', None)
        self._permxy = misc.get_default(kwargs, 'permxy', False)
        self._flipxy = misc.get_default(kwargs, 'flipxy', False)
        self._rescale_range = misc.get_default(kwargs, 'rescale_range', None)
        self._unit_scale = misc.get_default(kwargs, 'unit_scale', False)
        self._noisestd = misc.get_default(kwargs, 'noisestd', 0.0)
        print('Fetcher kwargs: {}'.format(kwargs))

    def reset_index(self):
        self._curri = 0

    def next_batch(self, batch_size):
        # print('Getting batch N: {} curri: {}'.format(self._N, self._curri))
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
            batches = list(d[self._perm[inds]] for d in self._datasets)
        else:
            batches = list(d[inds] for d in self._datasets)

        points = batches[0]
        n = points.shape[1]
        transformed_sets = []
        # TODO: vectorize?
        for i in range(batch_size):
            iset = points[i]
            if self._subsamp is not None and self._subsamp < n:
                iperm = np.random.permutation(n)
                iset = iset[iperm[:self._subsamp], :]
            if self._noisestd > 0.0:
                iset += self._noisestd*np.random.randn(*iset.shape)
            if self._flipxy:
                iset[:, 0] = (2.0*(np.random.rand() > 0.5)-1.0) * iset[:, 0]
                iset[:, 1] = (2.0*(np.random.rand() > 0.5)-1.0) * iset[:, 1]
            if self._permxy and np.random.rand() > 0.5:
                iset = iset[:, [1, 0, 2]]
            if self._rescale_range is not None:
                rrng = np.array(self._rescale_range)
                try:
                    rmult = 1.0 + rrng*(2.0*np.random.random(len(rrng))-1.0)
                    iset = rmult*iset
                except TypeError:
                    rmult = 1.0 + rrng*(2.0*np.random.random()-1.0)
                    iset = rmult*iset
            if self._unit_scale:
                minv = np.min(iset)
                maxv = np.max(iset)
                iset = (iset-minv)/(maxv-minv)
            transformed_sets.append(iset)
        batches[0] = np.stack(transformed_sets, 0)
        if len(batches) == 1:
            return batches[0]
        return batches


class DatasetFetchers:

    def __init__(self, train, validation, test,
                 subsamp=1000, subsamp_valid=1000, subsamp_test=1000,
                 noisestd=0.1,
                 flipxy=True, permxy=True, rescale_range=[0.2, 0.2, 0.2]):
        self.train = BatchFetcher(
            *train, subsamp=subsamp, flipxy=flipxy, permxy=permxy,
            rescale_range=rescale_range, noisestd=noisestd)
        self.validation = BatchFetcher(
            *validation, subsamp=subsamp_valid, flipxy=flipxy, permxy=permxy,
            rescale_range=rescale_range, noisestd=noisestd)
        self.test = BatchFetcher(
            *test, subsamp=subsamp_test, loop_around=False)

    def reset_index(self):
        self.train.reset_index()
        self.validation.reset_index()
        self.test.reset_index()

    @property
    def dim(self):
        return self.train.dim


def generate_fetchers(subsamp=1000, subsamp_valid=1000, subsamp_test=1000,
                      noisestd=0.1,
                      flipxy=True, permxy=True, rescale_range=[0.2, 0.2, 0.2]):
    return lambda tr, va, ts: DatasetFetchers(
        tr, va, ts,
        subsamp=subsamp, subsamp_valid=subsamp_valid, subsamp_test=subsamp_test,
        flipxy=flipxy, permxy=permxy, rescale_range=rescale_range,
        noisestd=noisestd)
