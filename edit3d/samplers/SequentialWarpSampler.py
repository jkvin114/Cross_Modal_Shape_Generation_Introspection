from torch.utils.data import Sampler

from edit3d.multimodal import logger


class SequentialWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        self.data_source = data_source
        self.n_repeats = n_repeats
        logger.info("[SequentialWarpSampler] Expanded data size: %s", len(self))

    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = list(range(len(self.data_source)))
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)

    def __len__(self):
        return len(self.data_source) * self.n_repeats