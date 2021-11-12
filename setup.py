from setuptools import setup

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "imaris_ims_file_reader/_version.py"}

setup(use_scm_version=use_scm,
    name='imaris_ims_file_reader',
    description='Imaris *.ims file format',
    url='https://github.com/CBI-PITT/imaris_ims_file_reader',
    author='Alan M Watson',
    author_email='alan.watson@pitt.edu',
    license='BSD 2-clause',
    packages=['imaris_ims_file_reader'],
    install_requires=[
        'h5py>=3.5.0',
        'numpy>=1.21.3',
        'psutil>=5.8.0',
        'scikit-image>=0.18.3'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
