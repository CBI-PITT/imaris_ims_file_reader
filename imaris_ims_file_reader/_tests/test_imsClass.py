import os
import h5py
import pytest
from imaris_ims_file_reader.ims import ims, ims_reader
import numpy as np
import zarr


TEST_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'brain_crop3.ims'
)


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


def test_swmr_fallback_when_file_already_open_non_swmr():
    """Opening with swmr=True fails if the file is already open non-SWMR.

    The library should catch the OSError and retry with swmr=False.
    """
    # Open the file non-SWMR first (simulating a prior h5py handle)
    pre_open = h5py.File(TEST_FILE, 'r')
    try:
        reader = ims(TEST_FILE)
        assert reader.hf is not None
        assert isinstance(reader.TimePoints, int)
        reader.close()
    finally:
        pre_open.close()


def test_swmr_false_avoids_warning():
    """Opening with swmr=False should work without any SWMR warning."""
    reader = ims(TEST_FILE, swmr=False)
    assert reader.hf is not None
    assert isinstance(reader.Channels, int)
    reader.close()


def test_close_does_not_crash_on_failed_construction():
    """close() should not raise AttributeError when open() fails.

    If open() raises (e.g. bad path), self.hf is never set.  __del__
    calls close(), which must handle this gracefully.
    """
    reader = ims_reader.__new__(ims_reader)
    # Simulate the state after __init__ set attributes but open() failed
    reader.filePathComplete = '/nonexistent/file.ims'
    reader.write = False
    reader.verbose = False
    reader.swmr = True
    reader.hf = None
    reader.dataset = None
    # close() should not raise
    reader.close()

    
