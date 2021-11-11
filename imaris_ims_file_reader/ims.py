import functools
import glob
import itertools
import os
import pickle
import random
import shutil
import sys

import h5py
import numpy as np
from psutil import virtual_memory


class IMS:
    def __init__(self, file, ResolutionLevelLock=None, cache_location=None, mem_size=20, disk_size=2000):
        self.filePathComplete = file
        self.filePathBase = os.path.split(file)[0]
        self.fileName = os.path.split(file)[1]
        self.fileExtension = os.path.splitext(self.fileName)[1]
        if cache_location is None and mem_size is None:
            self.cache = None
        else:
            self.cache = True
        self.cache_location = cache_location
        self.disk_size = disk_size * 1e9
        self.mem_size = mem_size * 1e9
        self.memCache = {}
        self.cacheFiles = []
        self.metaData = {}
        self.ResolutionLevelLock = ResolutionLevelLock

        with h5py.File(file, 'r') as hf:
            data_set = hf['DataSet']
            resolution_0 = data_set['ResolutionLevel 0']
            time_point_0 = resolution_0['TimePoint 0']
            channel_0 = time_point_0['Channel 0']
            data = channel_0['Data']

            self.ResolutionLevels = len(data_set)
            self.TimePoints = len(resolution_0)
            self.Channels = len(time_point_0)

            self.resolution = (
                round(
                    (self.read_numerical_dataset_attr('ExtMax2') - self.read_numerical_dataset_attr('ExtMin2'))
                    / self.read_numerical_dataset_attr('Z'),
                    3),
                round(
                    (self.read_numerical_dataset_attr('ExtMax1') - self.read_numerical_dataset_attr('ExtMin1'))
                    / self.read_numerical_dataset_attr('Y'),
                    3),
                round(
                    (self.read_numerical_dataset_attr('ExtMax0') - self.read_numerical_dataset_attr('ExtMin0'))
                    / self.read_numerical_dataset_attr('X'),
                    3)
            )

            self.shape = (
                self.TimePoints,
                self.Channels,
                int(self.read_attribute('DataSetInfo/Image', 'Z')),
                int(self.read_attribute('DataSetInfo/Image', 'Y')),
                int(self.read_attribute('DataSetInfo/Image', 'X'))
            )

            self.chunks = (1, 1, data.chunks[0], data.chunks[1], data.chunks[2])
            self.ndim = len(self.shape)
            self.dtype = data.dtype
            self.shapeH5Array = data.shape

            for r, t, c in itertools.product(range(self.ResolutionLevels), range(self.TimePoints),
                                             range(self.Channels)):
                location_attr = self.location_generator(r, t, c, data='attrib')
                location_data = self.location_generator(r, t, c, data='data')

                # Collect attribute info
                self.metaData[r, t, c, 'shape'] = (
                    t + 1,
                    c + 1,
                    int(self.read_attribute(location_attr, 'ImageSizeZ')),
                    int(self.read_attribute(location_attr, 'ImageSizeY')),
                    int(self.read_attribute(location_attr, 'ImageSizeX'))
                )
                self.metaData[r, t, c, 'resolution'] = tuple(
                    [round(float((origShape / newShape) * origRes), 3) for origRes, origShape, newShape in
                     zip(self.resolution, self.shape[-3:], self.metaData[r, t, c, 'shape'][-3:])]
                )
                self.metaData[r, t, c, 'HistogramMax'] = int(float(self.read_attribute(location_attr, 'HistogramMax')))
                self.metaData[r, t, c, 'HistogramMin'] = int(float(self.read_attribute(location_attr, 'HistogramMin')))

                # Collect dataset info
                self.metaData[r, t, c, 'chunks'] = (
                1, 1, hf[location_data].chunks[0], hf[location_data].chunks[1], hf[location_data].chunks[2])
                self.metaData[r, t, c, 'shapeH5Array'] = hf[location_data].shape
                self.metaData[r, t, c, 'dtype'] = hf[location_data].dtype

        if isinstance(self.ResolutionLevelLock, int):
            self.shape = self.metaData[self.ResolutionLevelLock, t, c, 'shape']
            self.ndim = len(self.shape)
            self.chunks = self.metaData[self.ResolutionLevelLock, t, c, 'chunks']
            self.shapeH5Array = self.metaData[self.ResolutionLevelLock, t, c, 'shapeH5Array']
            self.resolution = self.metaData[self.ResolutionLevelLock, t, c, 'resolution']
            self.dtype = self.metaData[self.ResolutionLevelLock, t, c, 'dtype']

            # TODO: Should define a method to change the ResolutionLevelLock after class in initialized

    def __getitem__(self, key):
        """
        All ims class objects are represented as shape (TCZYX)
        An integer only slice will return the entire timepoint (T) data as a numpy array

        Any other variation on slice will be coerced to 5 dimensions and
        extract that array

        If a 6th dimensions is present in the slice, dim[0] is assumed to be the resolutionLevel
        this will be used when choosing which array to extract.  Otherwise ResolutionLevelLock
        will be obeyed.  If ResolutionLevelLock is == None - default resolution is 0 (full-res)
        and a slice of 5 or less dimensions will extract information from resolutionLevel 0.

        ResolutionLevelLock is used when building a multi-resolution series to load into napari
        This option enables a 5D slice to lock on to a specified resolution level.
        """

        original_key = key
        res = self.ResolutionLevelLock

        if not isinstance(key, slice) and not isinstance(key, int) and len(key) == 6:
            res = key[0]
            if res >= self.ResolutionLevels:
                raise ValueError('Layer is larger than the number of ResolutionLevels')
            key = tuple((x for x in key[1::]))

        # All slices will be converted to 5 dims and placed into a tuple
        if isinstance(key, slice):
            key = [key]

        if isinstance(key, int):
            key = [slice(key)]
        # Convert int/slice mix to a tuple of slices
        elif isinstance(key, tuple):
            key = tuple((slice(x) if isinstance(x, int) else x for x in key))

        key = list(key)
        while len(key) < 5:
            key.append(slice(None))
        key = tuple(key)

        slice_returned = self.get_slice(
            r=res if res is not None else 0,  # Force ResolutionLock of None to be 0 when slicing
            t=self.slice_fixer(key[0], 't', res=res),
            c=self.slice_fixer(key[1], 'c', res=res),
            z=self.slice_fixer(key[2], 'z', res=res),
            y=self.slice_fixer(key[3], 'y', res=res),
            x=self.slice_fixer(key[4], 'x', res=res)
        )
        return slice_returned

    def read_numerical_dataset_attr(self, attrib):
        return float(self.read_attribute('DataSetInfo/Image', attrib))

    def slice_fixer(self, slice_object, dim, res):
        """
        Converts slice.stop == None to the origional image dims
        dim = dimension.  should be str: r,t,c,z,y,x

        Always returns a fully filled slice object (ie NO None)

        Negative slice values are not implemented yet self[:-5]

        Slicing with lists (fancy) is not implemented yet self[[1,2,3]]
        """

        if res is None:
            res = 0

        dims = {'r': self.ResolutionLevels,
                't': self.TimePoints,
                'c': self.Channels,
                'z': self.metaData[(res, 0, 0, 'shape')][-3],
                'y': self.metaData[(res, 0, 0, 'shape')][-2],
                'x': self.metaData[(res, 0, 0, 'shape')][-1]
                }

        if (slice_object.stop is not None) and (slice_object.stop > dims[dim]):
            raise ValueError('The specified stop dimension "{}" in larger than the dimensions of the \
                                 origional image'.format(dim))
        if (slice_object.start is not None) and (slice_object.start > dims[dim]):
            raise ValueError('The specified start dimension "{}" in larger than the dimensions of the \
                                 origional image'.format(dim))

        if isinstance(slice_object.stop, int) and slice_object.start == None and slice_object.step == None:
            return slice(
                slice_object.stop,
                slice_object.stop + 1,
                1 if slice_object.step is None else slice_object.step
            )

        if slice_object == slice(None):
            return slice(0, dims[dim], 1)

        if slice_object.step is None:
            slice_object = slice(slice_object.start, slice_object.stop, 1)

        if slice_object.stop is None:
            slice_object = slice(
                slice_object.start,
                dims[dim],
                slice_object.step
            )

        # TODO: Need to reevaluate if this last statement is still required
        if isinstance(slice_object.stop, int) and slice_object.start is None:
            slice_object = slice(
                max(0, slice_object.stop - 1),
                slice_object.stop,
                slice_object.step
            )

        return slice_object

    @staticmethod
    def location_generator(r, t, c, data='data'):
        """
        Given R, T, C, this funtion will generate a path to data in an imaris file
        default data == 'data' the path will reference with array of data
        if data == 'attrib' the bath will reference the channel location where attributes are stored
        """

        location = 'DataSet/ResolutionLevel {}/TimePoint {}/Channel {}'.format(r, t, c)
        if data == 'data':
            location = location + '/Data'
        return location

    def read_attribute(self, location, attrib):
        """
        Location should be specified as a path:  for example
        'DataSet/ResolutionLevel 0/TimePoint 0/Channel 1'

        attrib is a string that defines the attribute to extract: for example
        'ImageSizeX'
        """
        with h5py.File(self.filePathComplete, 'r') as hf:
            return str(hf[location].attrs[attrib], encoding='ascii')

    def get_slice(self, r, t, c, z, y, x):
        """
        IMS stores 3D datasets ONLY with Resolution, Time, and Color as 'directory'
        structure writing HDF5.  Thus, data access can only happen across dims XYZ
        for a specific RTC.
        """

        # incomingSlices = (r,t,c,z,y,x)
        t_size = list(range(self.TimePoints)[t])
        c_size = list(range(self.Channels)[c])
        z_size = len(range(self.metaData[(r, 0, 0, 'shape')][-3])[z])
        y_size = len(range(self.metaData[(r, 0, 0, 'shape')][-2])[y])
        x_size = len(range(self.metaData[(r, 0, 0, 'shape')][-1])[x])

        output_array = np.zeros((len(t_size), len(c_size), z_size, y_size, x_size))

        with h5py.File(self.filePathComplete, 'r') as hf:
            for idxt, t in enumerate(t_size):
                for idxc, c in enumerate(c_size):
                    d_set_string = self.location_generator(r, t, c, data='data')
                    output_array[idxt, idxc, :, :, :] = hf[d_set_string][z, y, x]

        """
        Some issues here with the output of these arrays.  Napari sometimes expects
        3-dim arrays and sometimes 5-dim arrays which originates from the dask array input representing
        tczyx dimensions of the imaris file.  When os.environ["NAPARI_ASYNC"] = "1", squeezing
        the array to 3 dimensions works.  When ASYNC is off squeese does not work.
        Napari throws an error because it did not get a 3-dim array.
    
        Am I implementing slicing wrong?  or does napari have some inconsistency with the 
        dimensions of the arrays that it expects with different loading mechanisms if the 
        arrays have unused single dimensions.
    
        Currently "NAPARI_ASYNC" = '1' is set to one in the image loader
        Currently File/Preferences/Render Images Asynchronously must be turned on for this plugin to work
        """
        try:
            if os.environ["NAPARI_ASYNC"] == '1':
                return np.squeeze(output_array)
        except KeyError:
            pass

        return output_array
