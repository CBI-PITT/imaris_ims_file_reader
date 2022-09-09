# imaris-ims-file-reader

Imaris file format reader - *.ims



```python
pip install imaris-ims-file-reader
```

```python
from imaris_ims_file_reader.ims import ims

a = ims(myFile.ims)

# Slice a like a numpy array always with 5 axes to access the highest resolution - level 0 - (t,c,z,y,x)
a[0,0,5,:,:] # Time point 0, Channel 0, z-layer 5

# Slice in 6 axes to designate the desired resolution level to work with - 0 is default and the highest resolution
a[3,0,0,5,:,:] # Resolution Level 3, Time point 0, Channel 0, z-layer 5

print(a.ResolutionLevelLock)
print(a.ResolutionLevels)
print(a.TimePoints)
print(a.Channels)
print(a.shape)
print(a.chunks)
print(a.dtype)
print(a.ndim)

# A 'resolution lock' can be set when making the class which allows for 5 axis slicing that always extracts from that resoltion level
a = ims(myFile.ims,ResolutionLevelLock=3)

# Change ResolutionLevelLock after the class is open
a.change_resolution_lock(2)
print(a.ResolutionLevelLock)

# The 'squeeze_output' option returns arrays in their reduced form similar to a numpy array.  This is True by default to maintain behavior similar to numpy; however, some applications may benefit from predictably returning a 5 axis array.  For example, napari prefers to have outputs with the same number of axes as the input.
a = ims(myFile.ims)
print(a[0,0,0].shape)
#(1024,1024)

a = ims(myFile.ims, squeeze_output=False)
print(a[0,0,0].shape)
#(1,1,1,1024,1024)

#########################################################
###  Open the Imaris file as a Zarr Store (read only) ###
#########################################################
from imaris_ims_file_reader.ims import ims
import zarr

store = ims(myFile.ims,ResolutionLevelLock=2, aszarr=True)
print(store)
#<imaris_ims_file_reader.ims_zarr_store.ims_zarr_store object at 0x7f48965f9ac0>

# The store object is NOT a sliceable array, but it does have arrtibutes that describe what to expect after opening the store.
print(store.ResolutionLevelLock)
print(store.ResolutionLevels)
print(store.TimePoints)
print(store.Channels)
print(store.shape)
print(store.chunks)
print(store.dtype)
print(store.ndim)

zarray = zarr.open(store,mode='r')
print(store.shape)
print(store.chunks)
print(store.dtype)
print(store.ndim)

print(zarray[0,0,0].shape)
#(1024,1024)
```



#### Change Log:

##### v0.1.3:  

Class name has been changed to all lowercase ('ims') to be compatible with many other dependent applications.

##### v0.1.4:  

Bug Fix:  Issue #4, get_Volume_At_Specific_Resolution does not extract the desired time point and color

**v0.1.5:**

-Compatibility changes for Napari.  

-Default behaviour changed to always return a 5-dim array.  squeeze_output=True can be specified to remove all single dims by automatically calling np.squeeze on outputs.

**v0.1.6:**

-Return default behaviour back to squeeze_output=True so that the reader performance more like a normal numpy array.

**v0.1.7:**

-Add warnings when HistogramMax and HistogramMin values are not present in channel data.  This is an issue when writing time series with PyImarisWriter.  The absence of these values may cause compatibility issues with programs that use imaris-ims-file-reader.

**v0.1.8:**

-Changed resolution rounding behaviour to make resolution calculation on ResolutionLevels > 0 more accurate

-Added option 'resolution_decimal_places' which enables the user to choose the number of decimal places to round resolutions (default:3).  'None' will NOT round the output.

-Added a new ims convenience function.  This aims to be a drop in replacement with all previous versions of the library, but adds an 'aszarr' option.  If aszarr=True (default:False), the object returned is a zarr store.  zarr.open(store,mode='r') to interact with the array.
