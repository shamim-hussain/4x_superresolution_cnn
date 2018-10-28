# -*- coding: utf-8 -*-

import h5py as hpy
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from random import shuffle


class DataGen:
    def __init__(self, dbfile, pshape=(64,64), strides=(32,32),
                 dsx=4,  ds_method='linear', us_method='cubic'):
        self.dbfile = dbfile
        interp = {'linear':cv2.INTER_LINEAR,
                 'cubic':cv2.INTER_CUBIC,
                 'area':cv2.INTER_AREA}
        self.ds_method = interp[ds_method]
        self.us_method = interp[us_method]
        self.pshape = pshape
        self.strides = strides
        self.dsx = dsx
        
        self.db = None
        self.image_dict = None
        self.datasets = None
        self.patch_list = None
    
    def load_db(self):
        self.db = hpy.File(self.dbfile, mode='r')
        return self.db
    
    def load_images(self,datasets=None,show_prog=True):
        if datasets==None:
            datasets = self.datasets if self.datasets!=None else self.get_datasets()
            image_dict=self.image_dict = {}
        else:
            image_dict = {}
            
        db = self.db 
        for k,fname in enumerate(datasets):
            if show_prog and (k+1)%100==0:
                print('{:^4}//{:^4}'.format(k+1,len(db)))
            image_dict[fname]=db[fname][:]
                    
        return image_dict
    
    def get_datasets(self):
        db = self.db
        self.datasets=[]
        db.visit(lambda x: self.datasets.append(x) 
                                if isinstance(db[x],hpy.Dataset)
                                else None)
        return self.datasets
    
    def estimate_patchno(self, datasets):
        num_patches = 0
        for dsi in datasets:
            ds=self.db[dsi]
            num_patches += ((ds.shape[0]-self.pshape[0]+self.strides[0]-1)//self.strides[0]+1)*\
                            ((ds.shape[1]-self.pshape[1]+self.strides[1]-1)//self.strides[1]+1)
            
        return num_patches
    
    def get_patch_list(self, datasets=None, pshape=None, strides=None, shdict=None):
        if datasets==None:
            datasets = self.datasets if self.datasets!=None else self.get_datasets()
            patch_list=self.patch_list=[]
        else:
            patch_list = []
        
        if shdict==None:
            shdict=self.db
        if pshape == None:
            pshape=self.pshape
        if strides == None:
            strides=self.strides
        for dsi in datasets:
            img=shdict[dsi]
            ys_list=list(range(0,img.shape[0]-pshape[0],strides[0]))+[img.shape[0]-pshape[0]]
            xs_list=list(range(0, img.shape[1]-pshape[1], strides[1]))+[img.shape[1]-pshape[1]]
            for ys in ys_list:
                ye=ys+pshape[0]
                for xs in xs_list:
                   xe=xs+pshape[1]
                   patch_list.append([dsi, np.s_[ys:ye], np.s_[xs:xe]])
        return patch_list
            
    def get_patch(self, patch):
        ds,ys,xs=patch
        return self.db[ds][ys,xs]
    
    def patch_gen(self, patch_list, bsize, from_images=True, up_sample=False):
        if from_images:
            if self.image_dict==None:
                self.load_images()
            if up_sample:
                def get_patch(patch):
                    ds,ys,xs=patch
                    X= self.image_dict[ds][ys,xs,...]
#                    if X.shape[-1]!=3:
#                        X=np.tile(X[...,None],(1,1,3))
                    Y = cv2.resize(X, None, fx=1./self.dsx, fy=1./self.dsx,
                                   interpolation=self.ds_method)
                    Y = cv2.resize(Y, None, fx=self.dsx, fy=self.dsx,
                                   interpolation=self.us_method)
                    return (X,Y)
            else:
                def get_patch(patch):
                    ds,ys,xs=patch
                    X= self.image_dict[ds][ys,xs,...]
#                    if X.shape[-1]!=3:
#                        X=np.tile(X[...,None],(1,1,3))
                    Y = cv2.resize(X, None, fx=1./self.dsx, fy=1./self.dsx,
                                   interpolation=self.ds_method)
                    return (X,Y)
        else:
            if up_sample:
                def get_patch(patch):
                    ds,ys,xs=patch
                    X= self.db[ds][ys,xs,...]
#                    if X.shape[-1]!=3:
#                        X=np.tile(X[...,None],(1,1,3))
                    Y = cv2.resize(X, None, fx=1./self.dsx, fy=1./self.dsx, 
                                   interpolation=self.ds_method)
                    Y = cv2.resize(Y, None, fx=self.dsx, fy=self.dsx,
                                   interpolation=self.us_method)
                    return (X,Y)
            else:
                def get_patch(patch):
                    ds,ys,xs=patch
                    X= self.db[ds][ys,xs,...]
#                    if X.shape[-1]!=3:
#                        X=np.tile(X[...,None],(1,1,3))
                    Y = cv2.resize(X, None, fx=1./self.dsx, fy=1./self.dsx, 
                                   interpolation=self.ds_method)
                    return (X,Y)
        
        ln=len(patch_list)
        
        with ThreadPool(processes=4) as pool:
            shuffle(patch_list)
            task = pool.map_async(get_patch, patch_list[-bsize:])
            while True:
                shuffle(patch_list)
                for istart in range(0,ln,bsize):
                    iend=min(istart + bsize, ln)
                    tup = task.get()
                    X=np.stack(x[0] for x in tup)
                    Y=np.stack(x[1] for x in tup)
                    task = pool.map_async(get_patch, patch_list[istart:iend])
                    yield (Y,X)
                    
                
    def closedb(self):
        self.db.close()
        










