from collections import namedtuple
import numpy as np
from scipy.signal import find_peaks, correlate


Truck = namedtuple('Truck', ['name', 'time', 'speed', 'signals', 'peaks', 'weights'])


def create_truck(name, events, time_idx, speed_idx, signal_idx, weights=None):
    time    = events[time_idx]
    shifted = events[speed_idx]
    signals = events[signal_idx]
    speed   = compute_speed(time, shifted, distance=3)
    peaks   = locate_peaks(signals)
    truck   = Truck(name, time, speed, signals, peaks, weights)
    return truck


def locate_peaks(signals, height=0.2):
    signals = np.atleast_2d(signals)
    i = signals.max(axis=1).argmax()
    peaks, _ = find_peaks(signals[i], height)
    return peaks


def compute_speed(time, shifted, distance):
    corr = correlate(shifted[1], shifted[0])    
    shift = corr.argmax() - (shifted.shape[1] - 1)
    delta = np.mean(time[1:] - time[:-1])
    speed = distance / (shift*delta)
    return speed
