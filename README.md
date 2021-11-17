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

```



#### Change Log:

##### v0.1.3:  

Class name has been changed to all lowercase ('ims') to be compatible with many other dependent applications.
