import gc
import os
import sys
from imaris_ims_file_reader.ims import ims
import numpy as np
import pytest
import zarr


# tmp_path is a pytest fixture
def test(tmp_path='brain_crop3.ims'):
    
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),tmp_path)
    # Test whether a ims file can be opened
    imsClass = ims(path)
    
    # Do we have some of the right attributes
    assert isinstance(imsClass.TimePoints, int)
    assert isinstance(imsClass.Channels, int)
    assert isinstance(imsClass.ResolutionLevels, int)
    assert isinstance(imsClass.resolution, tuple)
    assert len(imsClass.resolution) == 3
    assert isinstance(imsClass.metaData,dict)
    
    # Can we extract a numpy array?
    array = imsClass[imsClass.ResolutionLevels-1,0,0,:,:,:]
    assert isinstance(array,np.ndarray)

    # Will it open as a zarr store?
    store = ims(path, aszarr=True)
    zarray = zarr.open(store)
    array = zarray[0,0,:,:,:]
    assert isinstance(array, np.ndarray)
    


def test_failed_open_raises_cleanly(tmp_path):
    # A file that is not HDF5 (e.g. an IMS3 .ims, which is a TIFF container)
    path = tmp_path / 'not_hdf5.ims'
    path.write_bytes(b'II*\x00' + b'\x00' * 60)
    unraisable = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with pytest.raises(OSError):
            ims(str(path))
        gc.collect()
    finally:
        sys.unraisablehook = old_hook
    # close() in __del__ must not fail on the half-constructed reader
    assert unraisable == []
