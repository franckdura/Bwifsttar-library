from collections import namedtuple
import numpy as np
from scipy.signal import find_peaks, correlate




def create_truck(name, events, time_idx, speed_idx, signal_idx, weights=None):
    Truck = namedtuple('Truck', ['name', 'time', 'speed', 'signals', 'peaks', 'weights'])

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

def get_senlis_weights_modified():
    import numpy as np

    swm = {
    ####Trafic####
    
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
    
    ####CALIBRATION####
    
    '2015-07-02-10-54-53-222': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-16-36-41-533': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-09-29-10-28-52-687': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-09-56-08-457': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-09-28-46-125': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-15-52-01-990': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-07-02-17-05-11-125': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-16-08-40-480': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-15-40-32-876': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-11-54-10-861': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-11-24-46-054': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-09-29-15-23-14-921': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-55-32-562': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-28-15-093': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-00-38-369': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-12-17-33-320': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-13-33-05-960': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-11-50-31-328': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-11-08-45-998': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    }
    return swm

#IMPLÉMENTATION DE LA FONCTION DE CHARGEMENT DE LA DATA DANS LES DOSSIERS TRAFIC ET CALIBRATION


def read_file(name):
    import numpy as np
    data = []
    with open(name, 'r') as f:
        for line in f:
            data.append(line.split())
    data = np.array(data, dtype=float).T  # matrix of shape (n_signals, n_samples)
    return data

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

def load_senlis_modified(selected=slice(3,None)):
    """
        Données : Numéro du capteur 
        Sortie : Liste de namedTuples de type truck
        
        Fonction qui permet de retourner la liste des namedTuple "truck" en fonction du capteur choisi
        dans les dossiers trafic et calibration
    
    """
    from Bwifsttar import extract_data,get_senlis_weights_modified

    senlis_weights_modified = get_senlis_weights_modified()
    traffic_trucks = extract_data('data/senlis/trafic/', 0, [1,2], selected, senlis_weights_modified)
    calibration_trucks = extract_data('data/senlis/calibration/', 0, [1,2], selected, senlis_weights_modified)
    
    return calibration_trucks,traffic_trucks

