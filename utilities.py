import os
import numpy as np
import logging
import fcntl
from time import sleep


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def allocate_experiment_id(log_dir, id_seq_filename='._exp_id_seq'):
    seq_fd_path = os.path.join(log_dir, id_seq_filename)
    if not os.path.exists(seq_fd_path):
        with open(seq_fd_path, 'w') as fout:
            fout.write('1')
    with open(seq_fd_path, 'r+') as seq_fd:
        has_lock = False
        retries = 10
        while not has_lock and retries > 0:
            try:
                fcntl.flock(seq_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                has_lock = True
            except OSError:
                retries -= 1
                sleep(1)
        if not has_lock:
            raise OSError(f'Error: Could not get experiment ID from {seq_fd_path} by locking the file')
        new_id = int(seq_fd.read())
        seq_fd.seek(0)
        seq_fd.write(str(new_id + 1))
        seq_fd.truncate()
        fcntl.flock(seq_fd, fcntl.LOCK_UN)
    return new_id


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    exp_id = allocate_experiment_id(log_dir)

    log_path = os.path.join(log_dir, "%04d.log" % exp_id)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def calculate_scalar(x):

    if x.ndim <= 2:
        axis = 0

    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    max = np.max(x, axis=axis)

    return mean, std, max


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def mean_absolute_error(output, target):

    return np.mean(np.abs(output - target))


def signal_aggregate_error(output, target):

    return np.abs(np.sum(output) - np.sum(target))/np.sum(target)