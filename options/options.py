import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        self.parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
        self.parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
        self.parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        self.parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
        self.parser.add_argument("--dataset_A", type=str, default="/Users/hanwei/Desktop/cycle_gan/dataset/train_data_A.txt", help="the path of dataset A")
        self.parser.add_argument("--dataset_B", type=str, default="/Users/hanwei/Desktop/cycle_gan/dataset/train_data_B.txt", help="the path of dataset B")
        self.parser.add_argument("--dataset_labels", type=str, default="/Users/hanwei/Desktop/cycle_gan/dataset/train_label.txt", help="the path of dataset label")
        self.opt = self.parser.parse_args()

        self.img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)

        