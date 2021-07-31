def gather_trucks(calibration_trucks,traffic_trucks=[]):
    """
        Données :
            - calibration_trucks : Camions de calibration (namedTuple Truck)
            - traffic_trucks : Camion de traffic (namedTuple Truck)
        Sorties :
            - list_infl : liste des lignes d'influence
            - list_speed : liste des vitesses
            - list_meters : liste des mètres
            - list_times : liste des temps
            - list_signals : Liste des signaux
        Fonction : Retourne différentes listes de paramètres de l'ensemble des camions des deux listes passées
        en paramètre
    """
    
    import numpy as np
    from scipy import signal
    from Bwifsttar import calibration_mt_reg
    
    trucks = [truck for truck in calibration_trucks if(len(truck.weights) == len(truck.peaks))]
    trucks_t = [truck for truck in traffic_trucks if(len(truck.weights) == len(truck.peaks))]

    list_speeds = []
    list_meters= []
    list_infl = []
    list_times = []
    list_signals=[]

    for i in range(len(trucks)):
        list_speeds.append(trucks[i].speed)
        meters= trucks[i].speed*trucks[i].time
        list_meters.append(signal.resample(meters,701))
        list_infl.append(calibration_mt_reg(trucks[i:i+1],l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95}))
        list_times.append(signal.resample(trucks[i].time,701))
        list_signals.append(signal.resample(trucks[i].signals,701))


    list_infl = np.array(list_infl)
    list_speeds = np.array(list_speeds)
    list_meters = np.array(list_meters)
    list_times = np.array(list_times)
    list_signals = np.array(list_signals)

    return list_infl,list_speeds,list_meters,list_times,list_signals




def get_func_li(list_speeds,list_infl):
    """
        Données :
            - list_speeds : liste des vitesses
            - list_infl : liste des LI
        Sorties :
            - Retourne la fonction d'interpolation LI/meters
        Fonction : permet d'obtenir la fonction interpolée LI/meters
    """
    from scipy.interpolate import interp1d
    
    return interp1d(list_speeds, list_infl, fill_value="extrapolate",axis=0)


def gather_trucks_well_chosen(calibration_trucks,traffic_trucks):
    """
        Données :
            - calibration_trucks : Camions de calibration (namedTuple Truck)
            - traffic_trucks : Camion de traffic (namedTuple Truck)
        Sorties :
            - list_infl : liste des lignes d'influence
            - list_speed : liste des vitesses
            - list_meters : liste des mètres
            - list_times : liste des temps
            - list_signals : Liste des signaux
        Fonction : Retourne différentes listes de paramètres de l'ensemble des camions des deux listes passées
        en paramètre. Plus précisément, tous les camions de calibration et tous les camions de traffic allant à plus de 23.9m/s
    """
    
    import numpy as np
    from scipy import signal
    from Bwifsttar import calibration_mt_reg
    
    trucks = [truck for truck in calibration_trucks if(len(truck.weights) == len(truck.peaks))]
    trucks_t = [truck for truck in traffic_trucks if(len(truck.weights) == len(truck.peaks) and truck.speed>23)]

    list_speeds = []
    list_meters= []
    list_infl = []
    list_times = []
    list_signals=[]

    for i in range(len(trucks)):
        list_speeds.append(trucks[i].speed)
        meters= trucks[i].speed*trucks[i].time
        list_meters.append(signal.resample(meters,701))
        list_infl.append(calibration_mt_reg(trucks[i:i+1],l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95}))
        list_times.append(signal.resample(trucks[i].time,701))
        list_signals.append(signal.resample(trucks[i].signals,701))


    for i in range(len(trucks_t)):
        list_speeds.append(trucks_t[i].speed)
        meters= trucks_t[i].speed*trucks_t[i].time
        list_meters.append(signal.resample(meters,701))
        list_infl.append(
            calibration_mt_reg(trucks_t[i:i+1],l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95}))
        list_times.append(signal.resample(trucks_t[i].time,701))
        list_signals.append(signal.resample(trucks_t[i].signals,701))


    list_infl = np.array(list_infl)
    list_speeds = np.array(list_speeds)
    list_meters = np.array(list_meters)
    list_times = np.array(list_times)
    list_signals = np.array(list_signals)

    return list_infl,list_speeds,list_meters,list_times,list_signals

def get_func_li_from_scratch(calibration_trucks,traffic_trucks,well_chosen=False):
    """
        Données :
            - list_speeds : liste des vitesses
            - list_infl : liste des LI
        Sorties :
            - Retourne la fonction d'interpolation LI/meters
        Fonction : permet d'obtenir la fonction interpolée LI/meters
    """
    from scipy.interpolate import interp1d
    from Bwifsttar import gather_trucks_well_chosen,gather_trucks
    
    if well_chosen :
        list_infl,list_speeds,list_meters,list_times,list_signals = gather_trucks_well_chosen(calibration_trucks,traffic_trucks)
    else:
        list_infl,list_speeds,list_meters,list_times,list_signals = gather_trucks(calibration_trucks)

    
    return interp1d(list_speeds, list_infl, fill_value="extrapolate",axis=0)



def interp_lin(v_min,v_max,L_min,L_max):
    """
        Données :
            - v_min : vitesse minimale
            - v_max : vitesse maximale
            - L_min : influence à vitesse minimale
            - L_max : influence à vitesse maximale
        Sorties :
            - Retourne la fonction d'interpolation de la LI avec les vitesses
            
    """
    def func_lin(v,t):
        return abs((v-v_max)*L_min+(v-v_min)*L_max)/(v_max-v_min)
    return func_lin

