# -*- coding: utf-8 -*-
"""
This file is part of ethomap.

ethomap is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethomap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethomap.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2016-2018 Jacob M. Graving <jgraving@gmail.com>
Licensed under GPLv3, see LICENSE for details
"""

import numpy as np
import multiprocessing

class Parallel:
    def __init__(self, n_jobs):

        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count()+n_jobs+1
        elif n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()

        self.n_jobs = n_jobs
        self.pool = multiprocessing.Pool(n_jobs)

    def __call__(self, job, arg, asarray=False):    

        processed = self.pool.map(job, arg)

        if asarray:
            processed = np.array(processed)

        return processed
    
    def close(self):

        self.pool.close()
        self.pool.terminate()
        self.pool.join()

def _morlet_conj_ft(omegas, omega0=5.0):
    """
    Returns a Fourier conjugate Morlet wavelet

    Conjugate wavelet is given by the formula:
        np.pi**(-0.25) * np.exp(-0.5 * (omegas-omega0)**2)

    Parameters:
    ===========
    omegas : int
        omegas to calculate wavelets for
    omega0 : float (default=5.0)
        Dimensionless omega0 parameter for wavelet transform
    Returns:
    ========
    ft_wavelet : ndarray
        array of Fourier conjugate wavelets
    """

    ft_wavelet = np.pi**(-0.25) * np.exp(-0.5 * (omegas - omega0)**2)

    return ft_wavelet


def _morlet_fft_convolution(X, freqs, scales, dtime, omega0=5.0):
    """
    Calculates a Morlet continuous wavelet transform
    for a given signal across a range of frequencies

    Parameters:
    ===========
    X : array_like, shape (n_samples)
        Signal of interest
    freqs : array_like, shape (n_freqs)
        A list of frequencies
    scales : array_like, shape (n_freqs)
        A list of scales
    omega0 : float
        Dimensionless omega0 parameter for wavelet transform
    dtime : float
        Change in time per sample. The inverse of the sampling frequency.

    Returns:
    ========
    X_new : ndarray, shape (n_samples, n_freqs)
        The transformed signal.
    """

    n_samples = X.shape[0]
    n_freqs = freqs.shape[0]

    # allocate memory for result
    X_new = np.zeros((n_freqs, n_samples), dtype=np.complex128)

    # Test whether to add an extra zero
    if np.mod(n_samples, 2) == 1:
        X = np.concatenate((X, np.zeros(1)))
        n_samples = X.shape[0]
        pad_test = True
    else:
        pad_test = False

    # zero pad the array
    # padding = (np.zeros((n_samples / 2)), X, np.zeros((n_samples / 2)))
    # X = np.concatenate(padding)
    X = np.pad(X, pad_width=n_samples // 2, mode='symmetric', reflect_type='odd' )
    n_padded = X.shape[0]

    # calculate the omega values
    omegas = np.arange(-n_padded // 2, n_padded // 2) / (n_padded * dtime)
    omegas *= 2 * np.pi

    # Fourier transform the padded signal
    X_hat = np.fft.fft(X)
    X_hat = np.fft.fftshift(X_hat)

    # Set index to remove the extra zero if added
    if pad_test:
        idx0 = (n_samples // 2)
        idx1 = (n_samples // 2 + n_samples - 1)
    else:
        idx0 = (n_samples // 2)
        idx1 = (n_samples // 2 + n_samples)

    # Perform the wavelet transform
    for idx, scale in enumerate(scales):

        # calculate the wavelet
        morlet = -omegas * scale
        morlet = _morlet_conj_ft(morlet, omega0)

        # convolve the wavelet
        convolved = np.fft.ifft(morlet * X_hat)
        convolved = convolved[idx0:idx1]  # remove zero padding

        # scale power to account for disproportionally
        # large wavelet response at low frequencies
        power_scale = np.pi**-0.25
        power_scale *= np.exp(0.25* (omega0 - np.sqrt(omega0**2 + 2))**2)
        power_scale /= np.sqrt(2 * scale)
        convolved = convolved * power_scale

        X_new[idx] = convolved

    return X_new


def _morlet_fft_convolution_parallel(feed_dict):

    return _morlet_fft_convolution(**feed_dict)


def wavelet_transform(X, n_freqs, fsample, fmin, fmax,
                      prob=True, omega0=5.0, log_scale=True, n_jobs=1):
    """
    Applies a Morlet continuous wavelet transform to a data set
    across a range of frequencies.

    This is an implementation of the continuous wavelet transform
    described in Berman et al. 2014 (see references),
    which adjusts for disproportionally large wavelet response
    at low frequencies and removes amplitude fluctuations by
    normalizing the power spectrum at each sample.

    Parameters:
    ===========
    X : array_like, shape (n_samples, n_features)
        Data to transform
    n_freqs : int
        Number of frequencies to consider from fmin to fmax (inclusive)
    fsample : float
        Sampling frequency of the data (in Hz)
    fmin : float
        Minimum frequency of interest for a wavelet transform (in Hz)
    fmax : float
        Maximum frequency of interest for the wavelet transform (in Hz)
        Typically the Nyquist frequency of the signal (0.5 * fsample).
    prob : bool (default = True)
        Whether to normalize the power such that each sample sums to one.
        This effectively removes amplitude fluctuations.
    log_scale : bool (default = True)
        Whether to sample the frequencies on a log scale.
    omega0 : float (default = 5.0)
        Dimensionless omega0 parameter for wavelet transform.
    n_jobs : int
        Number of jobs to use for performing the wavelet transform.
        If -1, all CPUs are used. If 1 is given, no parallel computing is
        used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        Thus for n_jobs = -2, all CPUs but one are used.

    Returns:
    ========
    power : ndarray, shape (n_samples)
        The total power for each row in X_new
    X_new : ndarray, shape (n_samples, n_features*n_freqs)
        Continuous wavelet transformed X

    References:
    ===========
    Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014).
    Mapping the stereotyped behaviour of freely moving fruit flies.
    Journal of The Royal Society Interface, 11(99), 20140672.

    Notes:
    ======
    Based on code from Gordon J. Berman et al.
    (https://github.com/gordonberman/MotionMapper)
    """

    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtime = 1. / fsample

    if log_scale:
        fmin_log2 = np.log(fmin) / np.log(2)
        fmax_log2 = np.log(fmax) / np.log(2)
        freqs = np.logspace(fmin_log2, fmax_log2,
                            n_freqs, base=2)
    else:
        freqs = np.linspace(fmin, fmax, n_freqs)

    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * freqs)


    feed_dicts = [{"X": feature,
                   "freqs": freqs,
                   "scales": scales,
                   "dtime": dtime,
                   "omega0": omega0}
                  for feature in X.T]

    if n_jobs != 1:
        pool = Parallel(n_jobs)
        convolved = pool(_morlet_fft_convolution_parallel, feed_dicts)
        pool.close()
    elif n_jobs == 1:
        convolved = list(map(_morlet_fft_convolution_parallel, feed_dicts))

    X_new = np.array(convolved)
    
    amp = np.abs(X_new)
    phase = np.angle(X_new)
    power = amp.sum(axis=1, keepdims=True)
    
    if prob:
        amp /= power
    
    return freqs, amp, phase

def interpolate(x, shape):
    idx = np.arange(0,x.shape[0]*2,2)
    xvals = np.arange(0,x.shape[0]*2,1)
    interpolation = np.interp(xvals, idx, x)
    return interpolation[:shape]  

class FrequencyAnalysis:
    
    def __init__(self, n_freqs, fsample, fmin, fmax):
        self.n_freqs = n_freqs
        self.fsample = fsample
        self.fmin = fmin
        self.fmax = fmax
    def __call__(self, X, n_jobs=1):
        if X.ndim is not 2:
            raise ValueError('X.ndim must be 2')
        if X.shape[1] > 2:
            raise ValueError('n_features must be 2')
        freqs, amp, phase = wavelet_transform(X,
                                              self.n_freqs,
                                              self.fsample,
                                              self.fmin,
                                              self.fmax,
                                              False,
                                              n_jobs=n_jobs)
        freq_idx = [np.argmax(amp[idx],0) for idx in range(amp.shape[0])]
        freqs = np.array([freqs[idx] for idx in freq_idx]).T
        phase = np.array([phase[idx, jdx, np.arange(X.shape[0])] for (idx,jdx) in enumerate(freq_idx)])
        amps = np.array([amp[idx, jdx, np.arange(X.shape[0])] for (idx,jdx) in enumerate(freq_idx)]).T

        phase1 = phase[:, ::2]
        phase2 = phase[:, 1::2]
        if X.shape[0] % 2 != 0:
            phase2 = np.concatenate([phase2[:, 0][...,np.newaxis], phase2], 1)        

        phase1 = np.stack([interpolate(phase1[idx], X.shape[0]) for idx in range(phase1.shape[0])]).T
        phase2 = np.stack([interpolate(phase2[idx], X.shape[0]) for idx in range(phase2.shape[0])]).T

        phase_idx = np.argmax([np.sum((phase1 + np.pi) * (X + X.min())), np.sum((phase2 + np.pi) * (X + X.min()))])
        phase = phase1 if phase_idx is 0 else phase2
        phase_difference = phase[:, 1] - phase[:, 0]
        return {'amplitude': amps, 'frequency':freqs, 'phase': phase, 'phase_diff': phase_difference}


class PhaseAna(object):
    """docstring for PhaseAna"""
    def __init__(self):
        pass
        # super(PhaseAna, self).__init__()



    def getPhase(self,s):
        
        sst    = s
        dt     = 1.0/100.

        mother = 'PAUL'
        n      = 100
        t      = np.arange(0,len(s))/100.
        s0     = 0.1
        dj     = 0.8
        j1     = 3
        pad    = 10
        wave, period, scale, coi = self.wavelet(sst, dt, pad, dj, s0, j1, mother)
        phase = self.get_phase(wave)
        
        return phase

    # Wavelet transform:
    def wavelet(self,Y, dt, pad=0, dj=-1, s0=-1, J1=-1, mother=-1, param=-1):
        # # WAVELET  1D Wavelet transform with optional significance testing
        #   wave, period, scale, coi = wavelet(Y, dt, pad, dj, s0, J1, mother, param)
        #
        #   Computes the wavelet transform of the vector Y (length N),
        #   with sampling rate DT.
        #
        #   By default, the Morlet wavelet (k0=6) is used.
        #   The wavelet basis is normalized to have total energy=1 at all scales.
        #
        # INPUTS:
        #
        #    Y = the time series of length N.
        #    DT = amount of time between each Y value, i.e. the sampling time.
        #
        # OUTPUTS:
        #
        #    WAVE is the WAVELET transform of Y. This is a complex array
        #    of dimensions (N,J1+1). FLOAT(WAVE) gives the WAVELET amplitude,
        #    ATAN(IMAGINARY(WAVE),FLOAT(WAVE) gives the WAVELET phase.
        #    The WAVELET power spectrum is ABS(WAVE)**2.
        #    Its units are sigma**2 (the time series variance).
        #
        # OPTIONAL INPUTS:
        #
        # *** Note *** if none of the optional variables is set up, then the program
        #   uses default values of -1.
        #
        #    PAD = if set to 1 (default is 0), pad time series with enough zeroes to get
        #         N up to the next higher power of 2. This prevents wraparound
        #         from the end of the time series to the beginning, and also
        #         speeds up the FFT's used to do the wavelet transform.
        #         This will not eliminate all edge effects (see COI below).
        #
        #    DJ = the spacing between discrete scales. Default is 0.25.
        #         A smaller # will give better scale resolution, but be slower to plot.
        #
        #    S0 = the smallest scale of the wavelet.  Default is 2*DT.
        #
        #    J1 = the # of scales minus one. Scales range from S0 up to S0*2**(J1*DJ),
        #        to give a total of (J1+1) scales. Default is J1 = (LOG2(N DT/S0))/DJ.
        #
        #    MOTHER = the mother wavelet function.
        #             The choices are 'MORLET', 'PAUL', or 'DOG'
        #
        #    PARAM = the mother wavelet parameter.
        #            For 'MORLET' this is k0 (wavenumber), default is 6.
        #            For 'PAUL' this is m (order), default is 4.
        #            For 'DOG' this is m (m-th derivative), default is 2.
        #
        #
        # OPTIONAL OUTPUTS:
        #
        #    PERIOD = the vector of "Fourier" periods (in time units) that corresponds
        #           to the SCALEs.
        #
        #    SCALE = the vector of scale indices, given by S0*2**(j*DJ), j=0...J1
        #            where J1+1 is the total # of scales.
        #
        #    COI = if specified, then return the Cone-of-Influence, which is a vector
        #        of N points that contains the maximum period of useful information
        #        at that particular time.
        #        Periods greater than this are subject to edge effects.

        n1 = len(Y)

        if s0 == -1:
            s0 = 2 * dt
        if dj == -1:
            dj = 1. / 4.
        if J1 == -1:
            J1 = np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj)
        if mother == -1:
            mother = 'MORLET'

        #....construct time series to analyze, pad if necessary
        x = Y - np.mean(Y)
        if pad == 1:
            base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)  # power of 2 nearest to N
            x = np.concatenate((x, np.zeros((2 ** (base2 + 1) - n1).astype(np.int64))))

        n = len(x)

        #....construct wavenumber array used in transform [Eqn(5)]
        kplus = np.arange(1, np.fix(n / 2 + 1))
        kplus = (kplus * 2 * np.pi / (n * dt))
        kminus = (-(kplus[0:-1])[::-1])
        k = np.concatenate(([0.], kplus, kminus))

        #....compute FFT of the (padded) time series
        f = np.fft.fft(x)  # [Eqn(3)]

        #....construct SCALE array & empty PERIOD & WAVE arrays
        j = np.arange(0, J1+1)
        scale = s0 * 2. ** (j * dj)
        wave = np.zeros(shape=(int(J1 + 1), n), dtype=complex)  # define the wavelet array

        # loop through all scales and compute transform
        for a1 in range(0, int(J1+1)):
            daughter, fourier_factor, coi, dofmin = self.wave_bases(mother, k, scale[a1], param)
            wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]

        period = fourier_factor * scale  # [Table(1)]
        coi = coi * dt * np.concatenate((np.insert(np.arange(int((n1 + 1) / 2 - 1)), [0], [1E-5]),
            np.insert(np.flipud(np.arange(0, n1 / 2 - 1)), [-1], [1E-5])))  # COI [Sec.3g]
        wave = wave[:, :n1]  # get rid of padding before returning

        return wave, period, scale, coi

    def wave_bases(self,mother, k, scale, param):
        # WAVE_BASES  1D Wavelet functions Morlet, Paul, or DOG
        #
        #  DAUGHTER,FOURIER_FACTOR,COI,DOFMIN = wave_bases(MOTHER,K,SCALE,PARAM)
        #
        #   Computes the wavelet function as a function of Fourier frequency,
        #   used for the wavelet transform in Fourier space.
        #   (This program is called automatically by WAVELET)
        #
        # INPUTS:
        #
        #    MOTHER = a string, equal to 'MORLET' or 'PAUL' or 'DOG'
        #    K = a vector, the Fourier frequencies at which to calculate the wavelet
        #    SCALE = a number, the wavelet scale
        #    PARAM = the nondimensional parameter for the wavelet function
        #
        # OUTPUTS:
        #
        #    DAUGHTER = a vector, the wavelet function
        #    FOURIER_FACTOR = the ratio of Fourier period to scale
        #    COI = a number, the cone-of-influence size at the scale
        #    DOFMIN = a number, degrees of freedom for each point in the wavelet power
        #             (either 2 for Morlet and Paul, or 1 for the DOG)
        n = len(k)
        kplus = np.array(k > 0., dtype=float)

        if mother == 'MORLET':  # -----------------------------------  Morlet

            if param == -1:
                param = 6.

            k0 = np.copy(param)
            expnt = -(scale * k - k0) ** 2 / 2. * kplus
            norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * \
                    np.sqrt(n)  # total energy=N   [Eqn(7)]
            daughter = norm * np.exp(expnt)
            daughter = daughter * kplus  # Heaviside step function
            fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2)
                                                )  # Scale-->Fourier [Sec.3h]
            coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
            dofmin = 2  # Degrees of freedom
        elif mother == 'PAUL':  # --------------------------------  Paul
            if param == -1:
                param = 4.
            m = param
            expnt = -scale * k * kplus
            norm = np.sqrt(scale * k[1]) * (2 ** m / np.sqrt(m *
                np.prod(np.arange(1, (2 * m))))) * np.sqrt(n)
            daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
            fourier_factor = 4 * np.pi / (2 * m + 1)
            coi = fourier_factor * np.sqrt(2)
            dofmin = 2
        elif mother == 'DOG':  # --------------------------------  DOG
            if param == -1:
                param = 2.
            m = param
            expnt = -(scale * k) ** 2 / 2.0
            norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
            daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
            fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
            coi = fourier_factor / np.sqrt(2)
            dofmin = 1
        else:
            print('Mother must be one of MORLET, PAUL, DOG')

        return daughter, fourier_factor, coi, dofmin

    def get_phase(self,wave,method='median'):
        if method == 'median':
            aa = np.median(wave,axis=0)
        elif method == 'max':
            aa = np.max(wave,axis=0)
            ia = np.argmax(np.abs(wave)**2,axis=0)
            print(np.shape(wave))
            for ii,iia in enumerate(ia):
                aa[ii] = wave[:,iia]
        # aa = np.max(wave,axis=0)

        phase = np.angle(aa)

        phase = self.smooth_phase(phase)
        return phase

    def smooth_phase(self,phase):
    
        while len(phase[phase>np.pi])>0:
            phase[phase>np.pi] = phase[phase>np.pi] - 2*np.pi
        while len(phase[phase<-np.pi])>0:
            phase[phase<-np.pi] = phase[phase<-np.pi] + 2*np.pi

        for ii in range(len(phase)-1):
            if phase[ii+1] - phase [ii] > np.pi:
                phase[ii+1] = phase[ii+1] - 2*np.pi
            elif phase[ii+1] - phase[ii] < -np.pi:
                phase[ii+1] = phase[ii+1] +2*np.pi

        while np.mean(phase) > np.pi:
            phase = phase - 2*np.pi
        while np.mean(phase) < -np.pi:
            phase = phase + 2*np.pi
        return phase