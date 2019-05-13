import os
import numpy as np
import logging
import fcntl
from sklearn import metrics
from time import sleep
from datetime import datetime
import platform


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def binarize(tensor, threshold=0.5):
    return ((tensor - threshold) > 0).astype('float')


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
                fcntl.lockf(seq_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    return "%04d" % new_id


def allocate_experiment_id_alt(log_dir):
    return datetime.now().strftime('%Y%m%d_%H%M%S-{}-{}').format(platform.node(), os.getpid())


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    exp_id = allocate_experiment_id_alt(log_dir)

    log_path = os.path.join(log_dir, "%s.log" % exp_id)
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


def tp_fn_fp_tn(output, target):
    tp = np.sum(output + target > 1.5)
    fn = np.sum(target - output > 0.5)
    fp = np.sum(output - target > 0.5)
    tn = np.sum(output + target < 0.5)
    return tp, fn, fp, tn


def precision(output, target):
    (tp, fn, fp, tn) = tp_fn_fp_tn(output, target)
    if (tp + fp) == 0:
        return 0
    else:
        return float(tp) / (tp + fp)


def recall(output, target):
    (tp, fn, fp, tn) = tp_fn_fp_tn(output, target)
    if (tp + fn) == 0:
        return 0
    else:
        return float(tp) / (tp + fn)


def f_value(prec, rec):
    if (prec + rec) == 0:
        return 0
    else:
        return 2 * prec * rec / (prec + rec)


def roc_auc(output, target):
    return metrics.roc_auc_score(target, output)


def average_precision(output, target):
    return metrics.average_precision_score(target, output)
