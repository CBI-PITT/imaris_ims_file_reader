from setuptools import setup

setup(
    name='imaris_ims_file_reader',
    version='0.1.0',
    description='Imaris *.ims file format',
    url='https://github.com/CBI-PITT/imaris_ims_file_reader',
    author='Alan M Watson',
    author_email='alan.watson@pitt.edu',
    license='BSD 2-clause',
    packages=['imaris_ims_file_reader'],
    install_requires=[
        'h5py>=3.5.0',
        'numpy>=1.21.3',
        'psutil>=5.8.0'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)