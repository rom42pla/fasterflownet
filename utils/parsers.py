import argparse
import logging
import multiprocessing
from os.path import join, isdir

import torch


def parse_train_args():
    parser = argparse.ArgumentParser(description='Train FasterFlowNet')
    parser.add_argument("dataset", type=str,
                        help="The dataset to use for training or fine-tuning")
    parser.add_argument("--data_path", type=str, default=join(".", "data"),
                        help="where the datasets are kept")
    parser.add_argument("--weights_path", type=str, default=join(".", "weights"),
                        help="where the weights are kept")
    parser.add_argument("--device", type=str, default="auto",
                        help="the device where the model is trained (cpu or gpu)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--precision", type=int, default=16,
                        help="floating point precision (16 or 32 bit)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="number of workers for the dataloader")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of couple of frames in each batch")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--log_level", type=str, default="info",
                        help="logging level")
    args = parser.parse_args()

    assert args.dataset in {"sintel_final", "sintel_clean"}

    assert isdir(args.data_path)
    assert isdir(args.weights_path)
    assert args.device in {"cpu", "gpu", "cuda", "auto"}
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device in {"gpu", "cuda"}:
        args.device = "cuda"

    assert args.epochs >= 1
    assert args.precision in {16, 32}
    if args.num_workers is None:
        # sets num_workers as the CPU's number of cores
        args.num_workers = multiprocessing.cpu_count()
    else:
        assert args.num_workers >= 1
    assert args.batch_size >= 1
    assert args.log_level in {"debug", "info", "warning"}
    if args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == "info":
        logging.basicConfig(level=logging.INFO)
    if args.log_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    return args

def parse_test_args():
    parser = argparse.ArgumentParser(description='Test FasterFlowNet')
    parser.add_argument("dataset", type=str,
                        help="The dataset to use for training or fine-tuning")
    parser.add_argument("--data_path", type=str, default=join(".", "data"),
                        help="where the datasets are kept")
    parser.add_argument("--weights_path", type=str, default=join(".", "weights"),
                        help="where the weights are kept")
    parser.add_argument("--device", type=str, default="auto",
                        help="the device where the model is trained (cpu or gpu)")
    parser.add_argument("--precision", type=int, default=16,
                        help="floating point precision (16 or 32 bit)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="number of workers for the dataloader")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of couple of frames in each batch")
    parser.add_argument("--log_level", type=str, default="info",
                        help="logging level")
    args = parser.parse_args()

    assert args.dataset in {"sintel_final", "sintel_clean"}

    assert isdir(args.data_path)
    assert isdir(args.weights_path)
    assert args.device in {"cpu", "gpu", "cuda", "auto"}
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device in {"gpu", "cuda"}:
        args.device = "cuda"

    assert args.precision in {16, 32}
    if args.num_workers is None:
        # sets num_workers as the CPU's number of cores
        args.num_workers = multiprocessing.cpu_count()
    else:
        assert args.num_workers >= 1
    assert args.batch_size >= 1
    assert args.log_level in {"debug", "info", "warning"}
    if args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    elif args.log_level == "info":
        logging.basicConfig(level=logging.INFO)
    if args.log_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    return args