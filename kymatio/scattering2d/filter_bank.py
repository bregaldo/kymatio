"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

import os
import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.fft import fft2, ifft2

# Internal function for the parallel pre-building of the bandpass filters
def _build_bp_para(theta_list, M, N, j, L):
    ret = []
    for theta in theta_list:
        ret.append(morlet_2d(M, N, 0.8 * 2**j, (L // 2 - theta) * np.pi / L, 3.0 / 4.0 * np.pi /2**j, 4.0/L))
    return ret


def filter_bank(M, N, J, L=8, cplx=False):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []

    for j in range(J):
        # Parallel pre-build
        build_bp_para_loc = partial(_build_bp_para, M=N, N=N, j=j, L=L)
        nb_processes = os.cpu_count()
        work_list = np.array_split(np.arange(L), nb_processes)
        pool = mp.Pool(processes=nb_processes)
        results = pool.map(build_bp_para_loc, work_list)
        bp_filters = []
        for i in range(len(results)):
            bp_filters += results[i]
        pool.close()
        
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = bp_filters[theta]
            psi_signal_fourier = fft2(psi_signal)
            # drop the imaginary part, it is zero anyway
            psi_signal_fourier = np.real(psi_signal_fourier)
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_signal_fourier_res = periodize_filter_fft(
                    psi_signal_fourier, res)
                psi[res] = psi_signal_fourier_res
            filters['psi'].append(psi)
        if cplx:
            for theta in range(L):
                psi = {}
                psi['j'] = j
                psi['theta'] = theta + L
                for res in range(min(j + 1, max(J - 1, 1))):
                    psi[res] = fft2(np.conj(ifft2(filters['psi'][-L][res]))).real
                filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = fft2(phi_signal)
    # drop the imaginary part, it is zero anyway
    phi_signal_fourier = np.real(phi_signal_fourier)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = periodize_filter_fft(phi_signal_fourier, res)
        filters['phi'][res] = phi_signal_fourier_res

    return filters


def periodize_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.

        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)
    
    y = x.reshape(2 ** res, M // 2 ** res, 2 ** res, N // 2 ** res)
    out = y.sum(axis=(0, 2))

    return out


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1]:
        for ey in [-2, -1, 0, 1]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * np.pi * sigma * sigma / slant)
    gab /= norm_factor

    return gab


__all__ = ['filter_bank']
