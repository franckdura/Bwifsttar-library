# -*- coding: utf-8 -*-
# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import patsy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from pathlib import Path


def read_file(name):
    data = []
    with open(name, 'r') as f:
        for line in f:
            data.append(line.split())
    T = pd.DataFrame(data).astype(float)
    print("Size:", T.shape, "-> Path:", name)
    return T


def locate_peaks(signals, height=0.2):
    i = signals.max(axis=1).argmax()
    peaks, _ = scipy.signal.find_peaks(signals[i], height)
    return peaks


def compute_speed(time, shifted, distance):
    corr = scipy.signal.correlate(shifted[1], shifted[0])    
    shift = corr.argmax() - (shifted.shape[1] - 1)
    delta = np.mean(time[1:]-time[:-1])
    speed = distance / (shift*delta)
    return speed


def extract_data(root_path, selected = [3,4,6,7], velocity = [1,2], distance=3):
    files = Path(root_path).glob('*.txt')
    trucks = []
    for name in files:
        events  = read_file(name)
        time    = events[0].values
        signals = events[selected].values.T # Transposed to make a matrix of shape (n_signals, n_samples)
        shifted = events[velocity].values.T # Transposed to make a matrix of shape (n_signals, n_samples)
        peaks   = locate_peaks(signals)
        speed   = compute_speed(time, shifted, distance)
        trucks.append((name.stem, time, signals, peaks, speed))
    return pd.DataFrame(trucks, columns=['id', 'time', 'events', 'peaks', 'speed'])


def get_signal(index, trucks, weight_dict=None):
    print(trucks.iloc[index]['id'])
    time, speed, peaks, events = trucks.iloc[index][['time', 'speed', 'peaks', 'events']]
    if weight_dict is None:
        weights = None
    else:
        weights = weight_dict[trucks.iloc[index]['id']]
    return (time, speed, peaks), events, weights


def show_signal(truck, events, figsize=(12,8)):
    time, speed, peaks = truck
    plt.figure(figsize=figsize)
    meters = time*speed
    plt.plot(meters, events.T, zorder=2)
    for p in peaks:
        plt.axvline(time[p]*speed, linestyle='--', color='k', alpha=0.1)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.title('Speed: '+str(np.round(speed * 3.6, decimals=2))+' km/h')
    plt.xlabel('Meters')
    plt.show()


def compare_weights(estimated, groundthruth):
    index = np.arange(len(estimated)) + 1
    bar_width = 0.35
    ver_shift = 1
    opacity = 0.8
    error = error = np.abs(estimated - groundthruth).sum()
    plt.figure(figsize=(17,7))
    plt.bar(index-bar_width/2, estimated,    bar_width, alpha=opacity, color='b', label='Pesées en marche')
    plt.bar(index+bar_width/2, groundthruth, bar_width, alpha=opacity, color='r', label='Pesées statiques')
    plt.title('Erreur total: {:2.2f} t'.format(error), fontsize=14)
    plt.xlabel('Essieu', fontsize=14)
    plt.ylabel('Poid', fontsize=14)
    plt.legend(fontsize=14)
    for x, y, z in zip(index, groundthruth, estimated):
        plt.text(x+bar_width/2, y-ver_shift, '%.2f' % y, fontsize=16, fontweight='bold', color='white', ha='center')
        plt.text(x-bar_width/2, z-ver_shift, '%.2f' % z, fontsize=16, fontweight='bold', color='white', ha='center')
    plt.show()


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


def time_to_meter_interpolation(influence, time, speed):
    meters = speed * time[:len(influence)]
    dist = meters.max()
    func = interp1d(meters, influence, fill_value="extrapolate")
    return func, dist


def time_to_meter_sampling(influence, time, speed):
    func, dist = influence
    meters = speed * time
    meters = meters[meters<=dist]
    return func(meters)


def calibration(truck, events, weights, length, l2_reg=None, tv_reg=None):
    time, speed, peaks = truck
    A, b = prepare_least_squares(events, peaks, weights, length)
    A, b = prepare_regularization(A, b, l2_reg, tv_reg)
    influence ,_, _, _ = np.linalg.lstsq(A, b, rcond=None)
    func, dist = time_to_meter_interpolation(influence, time, speed)
    return func, dist


def reconstruction(truck, weights, influence):
    time, speed, peaks = truck
    influence = time_to_meter_sampling(influence, time, speed)
    toeplitz = create_toeplitz(time.size, influence.size, peaks)
    T_matrix = np.sum(weights[:,None,None] * toeplitz, axis=0)
    reconstructed = T_matrix @ influence
    return reconstructed, influence


def estimation(truck, events, influence):
    time, speed, peaks = truck
    influence= time_to_meter_sampling(influence, time, speed)
    toeplitz = create_toeplitz(events.size, influence.size, peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, events, rcond=None)
    return w


def prepare_least_squares(events, peaks, weights, length):
    toeplitz = create_toeplitz(events.shape[-1], length, peaks)
    A = np.sum(weights[:,None,None] * toeplitz, axis=0)
    b = events
    if events.ndim == 2:
        A = np.tile(A, reps=(events.shape[0],1))
        b = np.concatenate(events)
    return A, b


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
    if l2_reg is not None:
        win = apodization(length, l2_reg['cutoff'], l2_reg['strength']*total)
        A = np.concatenate((A, win))
        b = np.concatenate((b, np.zeros(length)))
    if tv_reg is not None:
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


def spline_calibration(truck, events, weights, length, l2_reg=None, tv_reg=None, **kwargs):
    time, speed, peaks = truck
    A, b = prepare_least_squares(events, peaks, weights, length)
    A, b = prepare_regularization(A, b, l2_reg, tv_reg)
    rough_influence = extract_first_peak(events, peaks)
    spline_matrix, _,_ = spline_approximation(rough_influence, length, **kwargs)
    A = A @ spline_matrix
    spline_coefs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    influence = spline_matrix @ spline_coefs
    func, dist = time_to_meter_conversion(influence, time, speed)
    return func, length, dist
