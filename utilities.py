import os
import numpy as np
import logging


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na
    
    
def create_logging(log_dir, filemode):
    
    create_folder(log_dir)
    i1 = 0
    
    while os.path.isfile(os.path.join(log_dir, "%04d.log" % i1)):
        i1 += 1
        
    log_path = os.path.join(log_dir, "%04d.log" % i1)
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

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean
    
    
def mean_absolute_error(output, target):
    
    return np.mean(np.abs(output - target))