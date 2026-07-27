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
import numpy as np
import json
import itertools

try:
    from zarr.abc.store import Store
    from zarr.core.buffer.core import default_buffer_prototype

    ZARR_V3 = True
except ImportError:  # pragma: no cover - exercised only with zarr<3
    from zarr._storage.store import Store

    default_buffer_prototype = None
    ZARR_V3 = False

import imaris_ims_file_reader as ims

class ims_zarr_store(Store):
    """
    Zarr storage adapter for reading IMS files
    """
    supports_consolidated_metadata = False
    supports_deletes = False
    supports_listing = True
    supports_partial_writes = False
    supports_writes = False

    def __init__(self, ims_file, ResolutionLevelLock = 0, writeable=False, normalize_keys=True, verbose=False, mode='r'):
        if ZARR_V3:
            super().__init__(read_only=True)

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

    def __eq__(self, other):
        return (
            isinstance(other, type(self))
            and self.path == other.path
            and self.ResolutionLevelLock == other.ResolutionLevelLock
            and self.normalize_keys == other.normalize_keys
            and self.writeable == other.writeable
        )

    def with_read_only(self, read_only=True):
        if not read_only:
            raise NotImplementedError("ims_zarr_store is read-only")
        return type(self)(
            self.path,
            ResolutionLevelLock=self.ResolutionLevelLock,
            writeable=False,
            normalize_keys=self.normalize_keys,
            verbose=self.verbose,
        )
    
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
        key_split = self._chunk_key_to_indices(key)
        
        index = []
        for idx,key_idx in enumerate(key_split):
            Start = self.chunks[idx] * key_idx
            Stop = Start + self.chunks[idx]
            Stop = Stop if Stop < self.shape[idx] else self.shape[idx]
            index.append((Start,Stop))
        
        return index

    def _chunk_key_to_indices(self, key):
        key_split = key.split('.')
        if len(key_split) != self.ndim:
            raise ValueError(key)

        key_split = [int(x) for x in key_split]
        chunk_counts = [
            (shape + chunk - 1) // chunk
            for shape, chunk in zip(self.shape, self.chunks)
        ]
        for key_idx, chunk_count in zip(key_split, chunk_counts):
            if key_idx < 0 or key_idx >= chunk_count:
                raise ValueError(key)

        return key_split
    
    def _fromfile(self,index):
        if self.verbose: print(index)
        array = self.ims[
            self.ResolutionLevelLock,
            index[0][0]:index[0][1],
            index[1][0]:index[1][1],
            index[2][0]:index[2][1],
            index[3][0]:index[3][1],
            index[4][0]:index[4][1]
            ]
        if self.verbose: print(array.shape)
        if array.shape == self.chunks:
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
        dtype = np.dtype(self.dtype).str
        
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
        "dimension_separator": ".",
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
        
    
    
    def _get_value(self, key):
        
        if self.verbose:
            print('GET : {}'.format(key))
        
        dset = self._dset_from_dirStoreFilePath(key)
        
        if dset is None:
            raise KeyError(key)
        if dset == '.zarray':
            return self._get_zarray()

        index = self._get_pixel_index_from_key(dset)
        return self._fromfile(index)

    def __getitem__(self, key):
        try:
            return self._get_value(key)
        except (KeyError, ValueError, IndexError):
            raise KeyError(key)

    def _get_bytes(self, key):
        value = self._get_value(key)
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value).tobytes(order='C')
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        return bytes(value)

    @staticmethod
    def _apply_byte_range(data, byte_range):
        if byte_range is None:
            return data

        if hasattr(byte_range, 'start') and hasattr(byte_range, 'end'):
            return data[byte_range.start:byte_range.end]
        if hasattr(byte_range, 'offset'):
            return data[byte_range.offset:]
        if hasattr(byte_range, 'suffix'):
            return data[-byte_range.suffix:]

        start, end = byte_range
        return data[start:end]

    async def _get_zarr_v3(self, key, prototype=None, byte_range=None):
        if prototype is None:
            prototype = default_buffer_prototype()
        try:
            data = self._get_bytes(key)
        except (KeyError, ValueError, IndexError):
            return None

        data = self._apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    def get(self, key, default=None, *, prototype=None, byte_range=None):
        if ZARR_V3:
            return self._get_zarr_v3(key, prototype=prototype, byte_range=byte_range)

        try:
            return self[key]
        except KeyError:
            return default

    def get_partial_values(self, prototype, key_ranges=None):
        if key_ranges is None:
            key_ranges = prototype
            return [
                self._apply_byte_range(self._get_bytes(key), byte_range)
                if key in self else None
                for key, byte_range in key_ranges
            ]

        async def _get_partial_values_zarr_v3():
            return [
                await self._get_zarr_v3(key, prototype=prototype, byte_range=byte_range)
                for key, byte_range in key_ranges
            ]

        return _get_partial_values_zarr_v3()

    async def exists(self, key):
        return key in self

    async def set(self, key, value):
        raise NotImplementedError("ims_zarr_store is read-only")

    async def delete(self, key):
        raise NotImplementedError("ims_zarr_store is read-only")
        

    def __setitem__(self, key, value):
        
        # key = self._normalize_key(key)
        
        if self.verbose:
            print('SET : {}'.format(key))
        
        pass

    def __delitem__(self, key):
        
        '''
        Does not yet handle situation where directorystore path is provided
        as the key.
        '''
        
        
        if self.verbose > 1:
            print('__delitem__')
            print('DEL : {}'.format(key))
        
        pass

    def __contains__(self, key):
        
        if self.verbose > 1:
            print('__contains__')
            print('CON : {}'.format(key))
        
        dset = self._dset_from_dirStoreFilePath(key)
        
        
        
        if dset == '.zarray':
            return True
                
        if self.verbose > 1:
            print('Store does not contain {}'.format(key))
            
        if dset is None:
            return False

        try:
            self._chunk_key_to_indices(dset)
        except (TypeError, ValueError):
            return False

        return True
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if getattr(self, 'ims', None) is not None:
            self.ims.close()
            self.ims = None
        if ZARR_V3:
            super().close()
    
    
    def keys(self):
        if self.verbose > 1:
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
        if self.verbose > 1:
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
        if self.verbose > 1:
            print('__iter__')
        return self.keys()

    async def list(self):
        for key in self.keys():
            yield key

    async def list_prefix(self, prefix):
        for key in self.keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix):
        prefix = prefix.rstrip('/')
        prefix = prefix + '/' if prefix else ''

        seen = set()
        for key in self.keys():
            if not key.startswith(prefix):
                continue
            entry = key[len(prefix):].split('/')[0]
            if entry and entry not in seen:
                seen.add(entry)
                yield entry

    def __len__(self):
        if self.verbose > 1:
            print('__len__')
        return sum(1 for _ in self.keys())
