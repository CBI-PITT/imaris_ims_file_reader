import os
from imaris_ims_file_reader.ims import ims
import numpy as np


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
    
    # Can we extract a numpy array
    array = imsClass[imsClass.ResolutionLevels-1,0,0,:,:,:]
    assert isinstance(array,np.ndarray)
    
