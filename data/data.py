from numpy.lib import utils
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):

    def __init__(self, image_direction_file_A, image_direction_file_B,label_filepath_file):
        self.image_directions_A = self.read_txt_file(image_direction_file_A)
        self.image_directions_B = self.read_txt_file(image_direction_file_B)
        self.labels = self.read_txt_file(label_filepath_file)

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