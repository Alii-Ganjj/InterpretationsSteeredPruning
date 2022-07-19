import os
import sys
import logging
import time
import socket
import torch

from argparse import ArgumentParser
from Utils.allenNLP_tee_logger import TeeLogger


def prepare_global_logging(serialization_dir: str, file_friendly_logging: bool) -> logging.FileHandler:
    """
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serialization_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carriage returns
        (used to update progress bars on a single terminal line). This
        option is typically only used if you are running in an environment
        without a terminal.

    Returns
    -------
    ``logging.FileHandler``
        A logging file handler that can later be closed and removed from the global logger.
    """

    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    # Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, "stdout.log")
    sys.stdout = TeeLogger(std_out_file, # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

    return stdout_handler


def logging_args(args):
    args_items = vars(args)
    args_keys = list(args_items.keys())
    args_keys.sort()
    for arg in args_keys:
        logging.warning(str(arg) + ': ' + str(getattr(args, arg)))


def set_logging_settings(args, main_file):
    args.data_dir = './dataset/cifar10'
    args.stdout = './results'

    args.device = torch.device(f"cuda:{args.gpu_ids[0]}" if (torch.cuda.is_available()) else "cpu")
    if 'cuda' in args.device.type:
        args.num_workers = 32
    else:
        args.num_workers = 0
    device_type = args.device.type
    index = str(args.device.index) if (isinstance(args.device.index, int)) else ''
    args.stdout = os.path.join(args.stdout, os.path.basename(os.getcwd()), main_file,
                               time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") + socket.gethostname() + '_' +
                               device_type + index)
    args.checkpoint_dir = os.path.join(args.stdout, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=args.logging_level)
    prepare_global_logging(serialization_dir=args.stdout, file_friendly_logging=False)
    if args.verbose > 0:
        logging_args(args)
    return args


def add_common_args(parent_parser):
    # Files Information.
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--logging_level', default=logging.INFO, help='Options: logging.DEBUG, logging.INFO')

    # Data Parameters
    parser.add_argument('--seed_data', default=1, help='random seed used for partitioning the data.')
    parser.add_argument('--im_size', default=32, help='size for resizing the images.')
    parser.add_argument('--num_classes', default=10)
    return parser


