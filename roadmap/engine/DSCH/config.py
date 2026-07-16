import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50_tanh", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../data", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=120, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=32, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--scheduler", type=str, default="step", help="step/none")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--parallel-val", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="flickr", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=38, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=128, help="length of hashing binary")

    # special settings

    parser.add_argument("--momentum", type=float, default=0.7, help="momentum of sgd")

    ### Schedular
    parser.add_argument("--gamma", default=0.3, type=float, help="Learning rate reduction after tau epochs.")
    parser.add_argument("--tau", default=[1000], nargs="+", type=int, help="Step size before reducing learning rate.")

    ### SCHLoss
    parser.add_argument("--alpha", default=1.0, type=float, help="hyper-parameters α")
    parser.add_argument("--beta", default=1.0, type=float, help="hyper-parameters β")

    args = parser.parse_args()

    # mods
    args.n_epochs = 100
    # args.momentum = 0.9
    # args.batch_size = 128
    # args.optimizer = "adamw"
    # args.lr = 2e-6
    # args.wd = 4e-4
    # args.tau = [30]

    return args
