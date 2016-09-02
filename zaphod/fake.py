'''
some tools for creating fake galaxies!
'''

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

class FakeIFU(object):
    '''Fake IFU manufacturing for spectral fit testing

    Parameters
    ----------
    true_spectra : :obj:`np.array`
        datacube of true spectra (no noise added)

    true_params : :obj:`list`
        a bunch of arrays with a similar shape to

    K : :obj:`np.array`
        spectral variance/covariance matrix

    Attributes
    ----------
    spatial_shape : :obj:`np.array`
        the image shape of the datacube

    '''
    def __init__(self, true_params, param_names, true_spectra):
        self.true_params = true_params
        self.true_spectra = true_spectra

    @classmethod
    def SingleSFH(cls, logl, K, spec, spatial_shape=[74, 74], SNmax=100):
        '''Make an IFU with a single SFH subject to various levels of noise

        Parameters
        ----------
        logl : :obj:`np.array`
            log-wavelength array, passed to `__init__`

        K : :obj:`np.array`
            see above
        '''
        pass

        return cls()

    def make_datacube(seed=None):
        if seed:
            np.random.seed(seed)

        noise = np.random.multivariate_normal()
