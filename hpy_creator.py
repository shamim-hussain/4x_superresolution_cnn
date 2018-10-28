# -*- coding: utf-8 -*-
"""
This file saves the JPEG files from a root directory
to a HDF5 file for easier access
"""

from scipy.misc import imread
from pathlib import Path
import numpy as np
import h5py as hpy

root = Path(r'E:\ImagNet3k_Flickr2k')
dest = Path(r'E:\mix5k.h5')

reject = True
dtype = np.uint8

if dest.exists() and input('Database exists; REMOVE? (y/n)').lower()=='y':
    dest.unlink()

np.random.seed(7)
#num_imag = 400

min_size=(64,64)

lst_imags=list(root.rglob('*.jpeg'))+list(root.rglob('*.jpg'))
#np.random.shuffle(lst_imags)
#sel_imags=lst_imags[:num_imag]
#sel_imags.sort()
sel_imags=lst_imags

with hpy.File(str(dest), mode='w') as db:
    db.attrs['root'] = str(root)
    db.attrs['dest'] = str(dest)
    db.attrs['dtype'] = str(dtype)
    for k, img_file in enumerate(sel_imags):
        ds='{:0>5}'.format(k)
        
        img = imread(str(img_file))
        
        assert (img.dtype == dtype)
        
        if len(img.shape)<3:
            
            print('B&W WARNING : {} is bnw - has shape {}'.
                  format(img_file.stem, img.shape))
            if reject : 
                print('REJECTED')
                continue
            img = np.tile(img[...,None],(1,1,3))
            
        if img.shape[0]<min_size[0]:
            print('LOW HEIGHT WARNING : {} has height = {}'.
                  format(img_file.stem, img.shape[0]))
            if reject : 
                print('REJECTED')
                continue
            pw = (min_size[0]-img.shape[0])//2+1
            img = np.pad(img, [(pw,pw), (0,0),(0,0)], 'reflect')
            
        if img.shape[1]<min_size[1]:
            if reject : continue
            print('LOW WIDTH WARNING : {} has width = {}'.
                  format(img_file.stem, img.shape[1]))
            if reject : 
                print('REJECTED')
                continue
            pw = (min_size[1]-img.shape[1])//2+1
            img = np.pad(img, [(0,0), (pw,pw), (0,0)], 'reflect')
        
        db[ds]=img
        db[ds].attrs['img_path'] = str(img_file.relative_to(root))
        if not (k+1) % 10:
            print('Written', k+1, 'files...')
            
    print('Written', len(db), 'files...')

print('Done!!!')