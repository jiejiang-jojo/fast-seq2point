#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import shutil
from multiprocessing import Pool
from utilities import allocate_experiment_id

def call_to_id_allocator(x):
    return allocate_experiment_id('tests/log')

def test_allocate_expeirment_id():
    os.makedirs('tests/log', exist_ok=True)
    with Pool(10) as pool:
        results = pool.map(call_to_id_allocator, range(1000))
    with open('tests/log/._exp_id_seq') as fin:
        next_id = int(fin.read())
    shutil.rmtree('tests/log')
    assert next_id == 1001
    assert len(results) == len(set(results))
    assert min(results) == 1
    assert max(results) == 1000
