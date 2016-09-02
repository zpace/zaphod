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
        true SFH parameters, for later comparison to fits

        a bunch of arrays with the same shape in the final two
        axes as `true_spectra`

    K : :obj:`np.ndarray`
        spectral variance/covariance matrix

    SNmax : float, optional
        maximum signal to noise

        equal to maximum of diagonal of K divided by flux

    logl : :obj:`np.ndarray`
            log-wavelength array, passed to `__init__`

    Attributes
    ----------

    '''

    def __init__(self, true_params, param_names, true_spectra, K, SNmax, logl):
        self.true_params = true_params
        self.param_names = param_names
        self.true_spectra = true_spectra

        self.nl = self.true_spectra.shape[0]
        self.spatial_shape = self.true_spectra.shape[-2:]

        if len(K.shape) > 2:
            raise ValueError('invalid shape, only spectral covariance allowed')
        elif len(K.shape) == 2:
            self.K = K
        else:
            if len(K) > 1:
                self.K = np.diag(K)
            else:
                self.K = np.diag(K * np.ones(self.nl))

        self.K /= (np.max(np.diag(K)) / SNmax**2.) # scale K now

    # =====
    # classmethods
    # =====

    @classmethod
    def SingleSFH(cls, spec, true_params, param_names, spatial_shape=[74, 74],
                  **kwargs):
        '''Make an IFU with a single SFH subject to various levels of noise

        Parameters
        ----------

        spec : :obj:`np.ndarray`, optional
            single spectrum

        '''

        pass

        return cls(**kwargs)

    # =====
    # properties
    # =====

    @property
    def image_shape(self):
        return self.true_spectra.shape[-2:]

    # =====
    # methods
    # =====

    def make_datacube(noise_model=None, noise_model_params=None, seed=None):
        '''Make a mock datacube with given noise characteristics

        Parameters
        ----------

        Returns
        -------
        :obj:`np.ndarray`
            has same shape as `self.true_spectra`
        '''
        if seed:
            np.random.seed(seed)

        noise = np.random.multivariate_normal(
            np.zeros(self.nl), self.K, self.spatial_shape)
        noise = np.moveaxis(noise, source=[0, 1, 2], destination=[1, 2, 0])

        if not noise_model:
            noise_model = lambda _: return np.ones(self.spatial_shape)
            noise_model_params = ['_'] # dummy

        noise_scale = noise_model(**noise_model_params)[np.newaxis, ...]

        return self.true_spectra + (noise * noise_scale)

    # =====
    # staticmethods
    # =====
