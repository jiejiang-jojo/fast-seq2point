import numpy as np
import h5py


class DataGenerator(object):
    
    def __init__(self, hdf5_path, house_list, target_device):
        
        hf = h5py.File(hdf5_path, 'r')
        
        for house in house_list:
            
            aggregate = hf[house]['aggregate']
            target = hf[house]['target_device']
            
        hf.close()
            