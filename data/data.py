from numpy.lib import utils
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path 