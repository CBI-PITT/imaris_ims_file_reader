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

# 'squeeze_output' option returns arrays in their reduced form similar to a numpy array.  This is off by default to maintain compatibility with the rest of the imaris-ims-file-reader API.
a = ims(myFile.ims)
print(a[0,0,0].shape)
#(1,1,1,1024,1024)

a = ims(myFile.ims, squeeze_output=True)
print(a[0,0,0].shape)
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
