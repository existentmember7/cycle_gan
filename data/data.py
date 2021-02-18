from numpy.lib import utils
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):

    def __init__(self, _opt):
        self.opt = _opt
        self.image_directions_A = self.read_txt_file(self.opt.dataset_A)
        self.image_directions_B = self.read_txt_file(self.opt.dataset_B)
        self.labels = self.read_txt_file(self.opt.dataset_labels)

    def __getitem__(self, idx):
        image_pair = dict()
        # print(self.image_directions_A[idx])
        # print((self.opt.img_size, self.opt.img_size))
        # print(label)
        image_pair["A"] = cv2.resize(cv2.imread(self.image_directions_A[idx].split('\n')[0], cv2.IMREAD_UNCHANGED), (self.opt.img_size, self.opt.img_size))
        image_pair["B"] = cv2.resize(cv2.imread(self.image_directions_B[idx].split('\n')[0], cv2.IMREAD_UNCHANGED), (self.opt.img_size, self.opt.img_size))
        label = int(self.labels[idx].split('\n')[0])

        return image_pair, label

    def __len__(self):
        return len(self.labels)
    
    def read_txt_file(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        return lines