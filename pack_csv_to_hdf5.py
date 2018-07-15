import argparse
import pandas as pd
import numpy as np
import os
import h5py


def add(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    
    # Paths
    hdf5_file = os.path.join(workspace, 'data.h5')
    
    # Write data of houses to hdf5 file
    csv_names = os.listdir(dataset_dir)
    csv_names = sorted(csv_names)
    
    hf = h5py.File(hdf5_file, 'w')
    
    for csv_name in csv_names:
        
        house_name = os.path.splitext(csv_name)[0]
        csv_path = os.path.join(dataset_dir, csv_name)
        
        df = pd.read_csv(csv_path, sep=',')
        df = pd.DataFrame(df)
        
        hf.create_group(house_name)

        if 'Unix' in df.keys():
            hf[house_name].create_dataset(name='unix', 
                                          data=np.array(df['Unix']), 
                                          dtype=np.int64)
            
        if 'Aggregate' in df.keys():
            hf[house_name].create_dataset(name='aggregate', 
                                          data=np.array(df['Aggregate']), 
                                          dtype=np.float32)
            
        if 'fridgefreezer' in df.keys():
            hf[house_name].create_dataset(name='fridgefreezer', 
                                          data=np.array(df['fridgefreezer']), 
                                          dtype=np.float32)
            
        if 'microwave' in df.keys():
            hf[house_name].create_dataset(name='microwave', 
                                          data=np.array(df['microwave']), 
                                          dtype=np.float32)
            
        if 'washingmachine' in df.keys():
            hf[house_name].create_dataset(name='washingmachine', 
                                          data=np.array(df['washingmachine']), 
                                          dtype=np.float32)
            
        if 'dishwasher' in df.keys():
            hf[house_name].create_dataset(name='dishwasher', 
                                          data=np.array(df['dishwasher']), 
                                          dtype=np.float32)
            
        if 'kettle' in df.keys():
            hf[house_name].create_dataset(name='kettle', 
                                          data=np.array(df['kettle']), 
                                          dtype=np.float32)
            
        if 'Issues' in df.keys():
            hf[house_name].create_dataset(name='issues', 
                                          data=np.array(df['Issues']), 
                                          dtype=np.int32)

        print("House {} write to {}".format(house_name, hdf5_file))
            
    hf.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    
    args = parser.parse_args()
    
    add(args)