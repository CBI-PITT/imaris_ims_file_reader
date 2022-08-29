# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:29:42 2022

@author: awatson
"""

'''
A Zarr store that uses HDF5 as a containiner to shard chunks accross a single
axis.  The store is implemented similar to a directory store 
but on axis[-3] HDF5 files are written which contain
chunks cooresponding to the remainining axes.  If the shape of the 
the array are less than 3 axdes, the shards will be accross axis0

Example:
    array.shape = (1,1,200,10000,10000)
    /root/of/array/.zarray
    #Sharded h5 container at axis[-3]
    /root/of/array/0/0/4.hf
    
    4.hf contents:
        key:value
        0.0:bit-string
        0.1:bit-string
        4.6:bit-string
        ...
        ...
'''


import os
import h5py
import shutil
import time
import numpy as np
import json
import itertools

from zarr.errors import (
    MetadataError,
    BadCompressorError,
    ContainsArrayError,
    ContainsGroupError,
    FSPathExistNotDir,
    ReadOnlyError,
)

from numcodecs.abc import Codec
from numcodecs.compat import (
    ensure_bytes,
    ensure_text,
    ensure_contiguous_ndarray
)
# from numcodecs.registry import codec_registry


from zarr.util import (buffer_size, json_loads, nolock, normalize_chunks,
                       normalize_dimension_separator,
                       normalize_dtype, normalize_fill_value, normalize_order,
                       normalize_shape, normalize_storage_path, retry_call)

from zarr._storage.absstore import ABSStore  # noqa: F401

from zarr._storage.store import Store
import imaris_ims_file_reader as ims

class ims_zarr_store(Store):
    """
    Zarr storage adapter for reading IMS files
    """

    def __init__(self, ims_file, ResolutionLevelLock = 0, writeable=False, normalize_keys=True, verbose=True, mode='r'):

        # guard conditions
        assert os.path.splitext(ims_file)[-1].lower() == '.ims'
        # if os.path.exists(path) and not os.path.isdir(path):
        #     raise FSPathExistNotDir(path)

        self.path = ims_file
        self.ResolutionLevelLock = ResolutionLevelLock
        self.normalize_keys = normalize_keys
        self.verbose = verbose #bool or int >= 1
        self.writeable = writeable
        self._files = ['.zarray','.zgroup','.zattrs','.zmetadata']
        self.ims = self.open_ims()
        self.ResolutionLevels = self.ims.ResolutionLevels
        
        self.TimePoints = self.ims.TimePoints
        self.Channels = self.ims.Channels
        self.chunks = self.ims.chunks
        self.shape = self.ims.shape
        self.dtype = self.ims.dtype
        self.ndim = self.ims.ndim
    
    def open_ims(self):
        return ims.ims(self.path,
                            ResolutionLevelLock=self.ResolutionLevelLock,
                            write=self.writeable,squeeze_output=False)
        
        
    def _normalize_key(self, key):
        return key.lower() if self.normalize_keys else key
    
    def _get_pixel_index_from_key(self,key):
        '''
        Key is expected to be 5 dims
        Function returns a slice in pixel coordinates for the provided key
        '''
        key_split = key.split('.')
        key_split = [int(x) for x in key_split]
        
        index = []
        for idx,key_idx in enumerate(key_split):
            Start = self.chunks[idx] * key_idx
            Stop = Start + self.chunks[idx]
            Stop = Stop if Stop < self.shape[idx] else self.shape[idx]
            index.append((Start,Stop))
        
        return index
    
    def _fromfile(self,index):
        print(index)
        array = self.ims[
            self.ResolutionLevelLock,
            index[0][0]:index[0][1],
            index[1][0]:index[1][1],
            index[2][0]:index[2][1],
            index[3][0]:index[3][1],
            index[4][0]:index[4][1]
            ]
        print(array.shape)
        if array.shape == self.chunks:
            print(True)
            return array
        else:
            canvas = np.zeros(self.chunks,dtype=array.dtype)
            canvas[
                0:array.shape[0],
                0:array.shape[1],
                0:array.shape[2],
                0:array.shape[3],
                0:array.shape[4]
                ] = array
            return canvas
    
    
    def _get_zarray(self):
        
        if self.dtype == 'uint16':
            dtype = "<u2"
        elif self.dtype == 'uint8':
            dtype = "|u1"
        elif self.dtype == 'float32':
            dtype = "<f4"
        elif self.dtype == float:
            dtype = "<f8"
        
        zarray = {
        "chunks": [
            *self.chunks
        ],
        "compressor": None,
        "dtype": dtype,
        "fill_value": 0.0,
        "filters": None,
        "order": "C",
        "shape": [
            *self.shape
        ],
        "zarr_format": 2
        }
        return json.dumps(zarray, indent=2).encode('utf-8')
    
    
    def _tofile(self,key, data, file):
        """ Write data to a file
        """
        pass
    
    def _dset_from_dirStoreFilePath(self,key):
        '''
        filepath will include self.path + key ('0.1.2.3.4')
        Chunks will be sharded along the axis[-3] if the length is >= 3
        Otherwise chunks are sharded along axis 0.
        Key stored in the h5 file is the full key for each chunk ('0.1.2.3.4')
        '''
        
        _ , key = os.path.split(key)
        
        key = self._normalize_key(key)
        
        if key in self._files:
            if key=='.zarray':
                return '.zarray'
            else:
                return None
        else:
            return key
        
    
    
    def __getitem__(self, key):
        
        if self.verbose:
            print('GET : {}'.format(key))
        
        dset = self._dset_from_dirStoreFilePath(key)
        # print(file)
        # print(dset)
        
        try:
            if dset is None:
                raise KeyError(key)
            if dset == '.zarray':
                return self._get_zarray()
            else:
                index = self._get_pixel_index_from_key(dset)
                return self._fromfile(index)
        except:
            raise KeyError(key)
        

    def __setitem__(self, key, value):
        
        # key = self._normalize_key(key)
        
        if self.verbose:
            print('SET : {}'.format(key))
            # print('SET VALUE : {}'.format(value))
        
        pass

    def __delitem__(self, key):
        
        '''
        Does not yet handle situation where directorystore path is provided
        as the key.
        '''
        
        
        if self.verbose == 2:
            print('__delitem__')
            print('DEL : {}'.format(key))
        
        pass

    def __contains__(self, key):
        
        if self.verbose == 2:
            print('__contains__')
            print('CON : {}'.format(key))
        
        dset = self._dset_from_dirStoreFilePath(key)
        # print(file)
        # print(dset)
        
        
        
        if dset == '.zarray':
            return True
                
        if self.verbose == 2:
            print('Store does not contain {}'.format(key))
            
        if dset is None:
            return False
        
        return True
    
    def __enter__(self):
        return self
    
    
    def keys(self):
        if self.verbose == 2:
            print('keys')
        if os.path.exists(self.path):
            yield from self._keys_fast()
            
            
    def _keys_fast(self):
        '''
        This will inspect each h5 file and yield keys in the form of paths.
        
        The paths must be translated into h5_file, key using the function:
            self._dset_from_dirStoreFilePath
        
        Only returns relative paths to store
        '''
        if self.verbose == 2:
            print('_keys_fast')
        yield '.zarray'
        chunk_num = []
        for idx in range(5):
           tmp = self.shape[idx]//self.chunks[idx]
           tmp = tmp if self.shape[idx]%self.chunks[idx] == 0 else tmp+1
           chunk_num.append(tmp)
        
        for t,c,z,y,x in itertools.product(
                range(chunk_num[0]),
                range(chunk_num[1]),
                range(chunk_num[2]),
                range(chunk_num[3]),
                range(chunk_num[4])
                ):
            
            yield '{}.{}.{}.{}.{}'.format(t,c,z,y,x)


    def __iter__(self):
        if self.verbose == 2:
            print('__iter__')
        return self.keys()

    def __len__(self):
        if self.verbose == 2:
            print('__len__')
        return len(self.keys())

