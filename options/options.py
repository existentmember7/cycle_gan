import argparse

class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
        self.parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        self.parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
        self.parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        self.parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--img_height", type=int, default=256, help="size of image height")
        self.parser.add_argument("--img_width", type=int, default=256, help="size of image width")
        self.parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        self.parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
        self.parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
        self.parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
        self.parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
        self.parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
        self.parser.add_argument("--dataset_A", type=str, default="/media/han/D/cy/dataset/train_data_A.txt", help="the path of dataset A")
        self.parser.add_argument("--dataset_B", type=str, default="/media/han/D/cy/dataset/train_data_B.txt", help="the path of dataset B")
        self.parser.add_argument("--dataset_labels", type=str, default="/media/han/D/cy/dataset/train_label.txt", help="the path of dataset label")
        self.opt = self.parser.parse_args()

        self.img_shape = (self.opt.channels, self.opt.img_width, self.opt.img_height)

        