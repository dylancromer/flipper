from distutils.core import setup, Extension
import os


setup(name='flipper',
      version='0.1',
      description='Map manipulation tools',
      author='Sudeep Das, Tobias Marriage',
      packages=['flipper'],
      install_requires=['healpy',
                        'numpy',
                        'astropy',
                        'matplotlib',
                        'pyfits',
                        'scipy',
                        'astLib'],
      zip_safe=False)
