# -*- coding: utf-8 -*-
# +
import numpy as np
import scipy.signal
import patsy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from pathlib import Path


def create_toeplitz(out_size, in_size, centers):
    """
    INPUTS:
    out_size - length of the output
    in_size  - length of the input (must be odd)
    centers  - list of translation centers
    
    RETURNS:
    Array of shape (len(centers), out_size, in_size) that gathers the Toeplitz matrices
    
    >>> out_size = 10
    >>> in_size = 7
    >>> centers = [4]
    >>> T = create_toeplitz(out_size, in_size, centers)
    >>> print(T)
    [[[0. 0. 0. 0. 0. 0. 0.]
      [1. 0. 0. 0. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 1. 0. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 0. 0. 1.]
      [0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0.]]]
      
    =============
    
    h = np.concatenate([np.linspace(0,.99,100), [1], np.linspace(0.99,0,100)])
    out = 1000
    tau = [10, 150, 300, 800]
    toeplitz = create_toeplitz(out, h.size, tau)
    y = toeplitz.sum(axis=0) @ h
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(h)
    plt.title('h')
    plt.subplot(1,2,2)
    plt.plot(y)
    plt.title('y = sum(toeplitz @ h)')
    plt.grid()
    plt.show()
    """
    toeplitz = []
    for k in centers:
        offset = in_size//2 - k
        D = np.ones((out_size, in_size))
        D = np.tril(D, offset)
        D = np.triu(D, offset)
        toeplitz.append(D)
    return np.stack(toeplitz, axis=0)


def time_to_meter_interpolation(truck, influence): 
    meters = truck.speed * truck.time[:len(influence)]#metres parcourus sur le temps len(influence)
    dist   = meters.max()#le nombre de metres total
    func   = interp1d(meters, influence, fill_value="extrapolate")#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)
    return func, dist#retourne la distance totale et la fonction d'interpolation


def time_to_meter_sampling(truck, influence_bundle):
    func, dist = influence_bundle
    meters = truck.speed * truck.time
    meters = meters[meters<=dist]
    return func(meters)#donne les valeurs via fonction d'interpolation des 'influence' (temps) pour une distance maximum


def calibration(truck, length, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    A, b = prepare_least_squares(truck, length)#retourne T et y
    A, b = prepare_regularization(A, b, l2_reg, tv_reg)#Aucune régularization pour le moment
    influence ,_, _, _ = np.linalg.lstsq(A, b, rcond=None)# Retourne la solution des moindres carrés de ||y - Th|| avec h notre ligne d'influence
    influence_bundle = time_to_meter_interpolation(truck, influence)
    return influence_bundle


def reconstruction(truck, influence_bundle):
    influence = time_to_meter_sampling(truck, influence_bundle)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, truck.peaks)
    T_matrix  = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence


def estimation(truck, influence_bundle):
    influence= time_to_meter_sampling(truck, influence_bundle)
    toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
    return w


def prepare_least_squares(truck, length):
    shape = truck.signals.shape#prend les dimensions du signal du camion (y)
    toeplitz = create_toeplitz(shape[-1], length, truck.peaks) #Da 
    A = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)# A = Somme des wa*Da = T
    b = truck.signals # b = y (le signal)
    if len(shape) == 2:
        A = np.tile(A, reps=(shape[0],1))
        b = np.concatenate(b)
    return A, b # retourne le signal et A (correspondant à T dans le cours)


def apodization(length, alpha, amplitude=1):
    win = 1 - scipy.signal.tukey(length, alpha)
    win = amplitude * win
    win = np.sqrt(win)
    win = np.diag(win)
    return win


def prepare_regularization(A, b, l2_reg=None, tv_reg=None):
    # assert l2_reg is None or tv_reg is None, "Only one regularization must be selected"
    length = A.shape[1]
    total = A.shape[0]
    if l2_reg is not None: #régularisation l2
        win = apodization(length, l2_reg['cutoff'], l2_reg['strength']*total)
        A = np.concatenate((A, win))
        b = np.concatenate((b, np.zeros(length)))
    if tv_reg is not None:#suppression des oscillations
        win  = apodization(length-1, tv_reg['cutoff'], tv_reg['strength']*total)
        diff = np.diag(np.ones(length)) - np.diag(np.ones(length-1),k=1)
        diff = diff[:-1]
        diff = win @ diff
        A = np.concatenate((A, diff))
        b = np.concatenate((b, np.zeros(length-1)))
    return A, b


def spline_approximation(signal, length=None, **kwargs):
    if length is None:
        extra = 0
    else:
        extra = length - signal.size
    pre = extra // 2
    post = extra - pre
    bbox = [-pre, signal.size + post]
    x = np.arange(signal.size)
    if extra > 0:
        spl = UnivariateSpline(x, signal, bbox=bbox, **kwargs)
    else:
        spl = UnivariateSpline(x, signal, **kwargs)
    spline_knots = spl.get_knots().astype(int)
    spline_coefs = spl.get_coeffs()
    spline_matrix = patsy.bs(np.arange(*bbox), knots=spline_knots[1:-1], include_intercept=True)
    spline_signal = spline_matrix @ spline_coefs
    return spline_matrix, spline_coefs, spline_signal


def extract_first_peak(events, peaks):
    first = events[:peaks[0]]
    last  = first[::-1] 
    last *= first.max() / last.max()
    influence = np.concatenate((first,last))
    return influence


def spline_calibration(truck, length, l2_reg=None, tv_reg=None, **kwargs):
    A, b = prepare_least_squares(truck, length)
    A, b = prepare_regularization(A, b, l2_reg, tv_reg)
    rough_influence = extract_first_peak(truck.signals, truck.peaks)
    spline_matrix, _,_ = spline_approximation(rough_influence, length, **kwargs)
    A = A @ spline_matrix
    spline_coefs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    influence = spline_matrix @ spline_coefs
    influence_bundle = time_to_meter_interpolation(truck, influence)
    return influence_bundle


def spline_denoising(truck, **kwargs):
    from utils import Truck
    name, time, speed, signals, peaks, weights = truck
    _, _, spline_signals = spline_approximation(signals, **kwargs)
    spline_truck = Truck(name, time, speed, spline_signals, peaks, weights)
    return spline_truck
