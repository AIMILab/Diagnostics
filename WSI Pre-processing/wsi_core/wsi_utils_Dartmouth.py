import h5py
import numpy as np
import os, cv2
import pdb
import matplotlib.pyplot as plt



def savePatchIter_bag_hdf5(patch,i):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(patch.values())
    
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape
    head_tail = os.path.split(name) 
    work_directory_P = os.path.join(save_path, head_tail[1]) 
    if not os.path.exists(work_directory_P):
        os.mkdir(work_directory_P)			
        
    fn = "{0}/{1}_{2}.png".format(work_directory_P, head_tail[1], i)
    cv2.imwrite(fn, img_patch[0].astype(np.uint8))

    file_path = os.path.join(save_path, head_tail[1])+ '.h5'
    file = h5py.File(file_path, "a")

    dset = file['imgs']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    if 'coords' in file:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
        coord_dset[-img_shape[0]:] = (x,y)

    file.close()



def initialize_hdf5_bag(first_patch, save_coord=False):
    x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())
    head_tail = os.path.split(name) 
    
    i = 0
    work_directory_P = os.path.join(save_path, head_tail[1]) 
    if not os.path.exists(work_directory_P):
        os.mkdir(work_directory_P)			
        
    fn = "{0}/{1}_{2}.png".format(work_directory_P, head_tail[1], i)
    img_patch = np.array(img_patch)[np.newaxis,...]
    cv2.imwrite(fn, img_patch[0].astype(np.uint8))

    file_path = os.path.join(save_path, head_tail[1])+ '.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(img_patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs', shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    dset.attrs['patch_level'] = patch_level
    dset.attrs['wsi_name'] = name
    dset.attrs['downsample'] = downsample
    dset.attrs['level_dim'] = level_dim
    dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = (x,y)

    file.close()
    return file_path