#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: data_stats.py
Description: Summarize data
"""

import sys
from data_generator import binarize, DataGenerator

def binary_class_stats(data_generator, threshold=0.5):
    """ Check the balance of binary classes"""
    binary_data = binarize(data_generator.train_y, threshold)
    ones = binary_data.sum()
    zeros = binary_data.size - ones
    return {
        'ones': ones,
        'zeros': zeros,
        'percentage_of_ones': float(ones)/(ones + zeros)
    }

def stats(datafile, device, households, **kwargs):
    """ Generate stats
    :param datafile: hdf5 data file
    :param households: the list of names of the households
    :param device: the device name
    :param kwargs: the extra argument for different types of stats
    :returns: a dictionary of stats

    """
    data_generator = DataGenerator(datafile, device, households, households, 100, 127, 10)
    print(f'binary class ratio = {binary_class_stats(data_generator, **kwargs)}')


if __name__ == "__main__":
    stats(sys.argv[1], sys.argv[2], sys.argv[3].split(','), threshold=float(sys.argv[4]))
