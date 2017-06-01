flipper
=======

Originally authored by Sudeep Das and Toby Marriage, this fork of flipper is jointly maintained on the ACT github organization.

Documentation can be found at:
http://www.hep.anl.gov/sdas/flipperDocumentation/

In contrast with the original version, after installation (see below) you now import the flipper module you need from the flipper package. For example,

```
from flipper import liteMap
import flipper.fftTools as ft
```

This fork of flipper also includes a copy of Sigurd Naess' enlib.fft module, a wrapper for fast multi-threaded pyfftw FFTs.

Flipper is a light-weight python tool for working with CMB data which broadly provides three main functionalities:
A suite of tools for operations performed on maps, like application of filters, taking gradients etc.
An FFT and power spectrum tool, 
implementing the core concepts from Das, Hajian and Spergel (2008) http://arxiv.org/abs/0809.1092v1

Plus, Flipper has awesome visualization tools based on Matplotlib. 

Flipper has become one of the most heavily used base codes in the Atacama Cosmology Telescope collaboration 
and led to a slew of papers with over 300 citations. 


Dependencies
==============

The following dependencies will be installed if they are not found:

numpy 

scipy http://www.scipy.org/

pyfits http://www.stsci.edu/resources/software_hardware/pyfits

astLib http://astlib.sourceforge.net/

matplotlib http://matplotlib.sourceforge.net/

healpy http://code.google.com/p/healpy/


Installation
===============

1. Fork and clone this repository
2. In the repository root directory, run
```
pip install -e . --user
```

That's it! Now you should be able to import flipper modules as described earlier from anywhere on your system as long as `pip` is installed correctly.

The `-e` makes symbolic links instead of copying files, so don't delete the cloned repository after. Changes you make will be reflected immediately. You are encouraged to make useful changes and issue pull requests to help maintain this code.

