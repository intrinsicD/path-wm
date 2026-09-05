"""Episode-disjoint action-conditioned trajectory data."""
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, path, episodes, frameskip=5, num_steps=4):
        raise NotImplementedError

def split_episodes(count, seed=3072, train_fraction=0.9):
    raise NotImplementedError
