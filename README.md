# imaris-ims-file-reader

Imaris file format reader



```python
pip install imaris-ims-file-reader
```

```python
from imaris_ims_file_reader import ims

a = ims(myFile.ims)

# Slice a like a numpy array always with 5 axes (t,c,z,y,x)
a[0,0,5,:,:] # Time point 0, Channel 0, z-layer 5

```

