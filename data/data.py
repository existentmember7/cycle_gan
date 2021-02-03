from numpy.lib import utils
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):

    def __init__(self, _opt):
        self.opt = _opt
        self.image_directions_A = self.read_txt_file(self.opt.dataset_A)
        self.image_directions_B = self.read_txt_file(self.opt.datset_B)
        self.labels = self.read_txt_file(self.opt.dataset_labels)

    def __getitem__(self, idx):
        image_pair = dict()
        image_pair["A"] = cv2.imread(self.image_directions_A[idx])
        image_pair["B"] = cv2.imread(self.image_directions_B[idx])
        label = self.labels[idx]

        return image_pair, label

    def __len__(self):
        return(len(self.labels))
    
    def read_txt_file(self, filename):
        file = open(filename, "r")
        lines = file.readlines
        return lines