import numpy as np
from .utils import create_truck
import tensorflow as tf
import sklearn

senlis_weights = {
    'PL1_2015-10-01-09-45-52-576':  np.array([7.35, 11.25, 8.20, 8.10, 8.25]),
    'PL2_2015-10-01-09-55-36-371':  np.array([7.65, 14.10, 6.05, 6.20, 6.30]),
    'PL3_2015-10-01-10-22-34-654':  np.array([7.40, 11.60, 8.35, 8.50, 8.50]),
    'PL4_2015-10-01-10-35-43-279':  np.array([7.60, 12.35, 7.80, 7.70, 7.80]),
    'PL5_2015-10-01-10-59-39-060':  np.array([7.35, 10.50, 9.30, 9.70, 9.45]),
    'PL6_2015-10-01-11-05-36-548':  np.array([7.30, 14.30, 6.80, 6.60, 6.70]),
    'PL7_2015-10-01-11-14-06-712':  np.array([7.05, 13.95, 7.25, 7.55, 7.90]),
    'PL8_2015-10-01-11-42-22-222':  np.array([6.40, 14.05, 6.85, 6.75, 6.65]),
    'PL9_2015-10-01-14-42-22-908':  np.array([8.20, 12.00, 8.35, 8.40, 8.35]),
    'PL10_2015-10-01-14-58-50-212': np.array([6.75, 12.45, 8.00, 8.10, 8.00]),
    'PL11_2015-10-01-15-00-57-826': np.array([7.15, 11.40, 8.05, 8.00, 8.45]),
    'PL12_2015-10-01-15-10-30-906': np.array([7.40, 10.80, 8.80, 9.05, 8.75]),
    'PL13_2015-10-01-15-19-48-718': np.array([8.15, 12.20, 7.30, 8.30, 8.80]),
    'PL14_2015-10-01-15-34-03-982': np.array([6.20,  5.05, 7.95, 7.65, 7.40, 7.65]),
    'PL15_2015-10-01-15-36-13-519': np.array([5.35,  8.25, 8.10, 7.75, 7.80, 7.60]),
}


def load_senlis(selected=slice(3,None)):
    return extract_data('data/senlis/trafic/', 0, [1,2], selected, senlis_weights)


def extract_data(root_path, time_idx, speed_idx, signal_idx, static_weights=None):
    from pathlib import Path
    files = Path(root_path).glob('*.txt')
    trucks = []
    for name in files:
        events  = read_file(name)
        name    = name.stem
        weights = static_weights[name] if static_weights is not None else None
        truck   = create_truck(name, events, time_idx, speed_idx, signal_idx, weights)
        trucks.append(truck)
    return trucks


def read_file(name):
    data = []
    with open(name, 'r') as f:
        for line in f:
            data.append(line.split())
    data = np.array(data, dtype=float).T  # matrix of shape (n_signals, n_samples)
    return data
