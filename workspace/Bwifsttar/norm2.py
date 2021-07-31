
def create_truck_norm2(name, events, time_idx, speed_idx, signal_idx, static_weights=None):
    """
        Données :
            - name,events,time_idx,speed_idx,signal_idx,static_weights : caractéristiques du camion
        Sorties :
            - truck : Nouveau namedTuple avec intégration de la normalisation sur le signal
        Fonction : Crée un nouveau namedTuple avec intégration de la normalisation de l'échelle du signal
    """
    from collections import namedtuple
    from Bwifsttar import time_to_meter_conversion_mt_norm1
    from Bwifsttar import locate_peaks,compute_speed
    
    NewTruck = namedtuple('NewTruck', ['name','time', 'meters', 'speed','signals', 'peaks', 'weights'])

    time    = events[time_idx]
    shifted = events[speed_idx]
    signals = events[signal_idx]
    speed   = compute_speed(time, shifted, distance=3)
    peaks   = locate_peaks(signals)
    meters, signals = time_to_meter_conversion_mt_norm1(time, signals, speed, peaks)
    peaks   = locate_peaks(signals)
    truck   = NewTruck(name, time,meters,speed, signals, peaks, static_weights)
    return truck

def extract_data_norm2(root_path, time_idx, speed_idx, signal_idx, static_weights=None):
    from pathlib import Path
    from Bwifsttar import create_truck_norm2,read_file
    
    files = Path(root_path).glob('*.txt')
    trucks = []
    for name in files:
        events  = read_file(name)
        name    = name.stem
        weights = static_weights[name] if static_weights is not None else None
        truck   = create_truck_norm2(name, events, time_idx, speed_idx, signal_idx, weights)
        trucks.append(truck)
    return trucks

def load_senlis_modified_norm2(selected=slice(3,None)):
    """
        Données :
            - selected : Numéro identifiant le capteur
        Sorties :
            - calibration_trucks : Camions de calibration
            - traffic_trucks : Camions de traffic
        Fonction : Crée les listes de camions avec l'intégration de la normalisation du signal
    """
    from Bwifsttar import extract_data,get_senlis_weights_modified,extract_data_norm2
    
    senlis_weights_modified = get_senlis_weights_modified()
    traffic_trucks = extract_data_norm2('data/senlis/trafic/', 0, [1,2], selected, senlis_weights_modified)
    calibration_trucks = extract_data_norm2('data/senlis/calibration/', 0, [1,2], selected, senlis_weights_modified)
    
    return calibration_trucks,traffic_trucks

def calibration_mt_norm2(trucks,length, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    """
        Données :
            - trucks : liste des camions de calibration
            - length : Longueur de la liste d'influence
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation Totale Variance
        Sorties :
            - influence_finale.x : Ligne d'influence estimée à partir des camions de calibration
        Fonction : Calcul la ligne d'influence à partir d'une liste de camions
    """
    import numpy as np
    from Bwifsttar import get_std
    from scipy.optimize import minimize

    #Préparation des matrices utiles à la minimisation
    
    T_tilde, y_tilde = prepare_least_squares_mt(trucks,length)
    
    for T in T_tilde:
        T = T.astype(float)
        
    for y in y_tilde:
        y = y.astype(float)

    trucks_calculables = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    
    #Coeffs de pondération par niveau de bruit
    
    ws = np.array([])
    for truck in trucks_calculables:
        ws=np.append(ws,1/get_std(truck))
    ws = 0.001*ws
    
    #Définition de la ligne d'influence 0 (pour commencer le calcul)
    
    infl0,_, _, _ = np.linalg.lstsq(T_tilde[0],y_tilde[0], rcond=None)  
   
    #Définition de la fonction à minimiser 
    def func_finale_to_min(h):
        """
            Données : 
                - h : la ligne d'influence (que nous cherchons par la minimisation)
            Sorties :
                - sum_to_minimize : Somme des moindres carrés à minimiser
            Fonction :
                Prend en paramètre la LI et retourne la somme à minimiser
        """
        norm_array = np.array([])
        for i in range(T_tilde.shape[0]):
            #norm_array = np.append(norm_array,func_to_min(h,i))
            norm_array = np.append(norm_array,ws[i]*np.linalg.norm(T_tilde[i]@h-y_tilde[i])**2)
            
        sum_to_minimize = np.sum(norm_array)
        return sum_to_minimize
    

    influence_finale = minimize(func_finale_to_min,infl0,method='Nelder-Mead',tol=0.1)#utiliser CG pour plus rapidité
    
    return influence_finale.x

def time_to_meter_sampling_mt_norm2(truck, influence):
    """
        Données :
            - truck : camion (type namedTuple)
            - influence : Ligne d'influence 
        Sortie :
            - func(meters) : Ligne d'influence estimée sur le vecteur meters 
        Fonction : Retourne la ligne d'influence adaptée aux meters du camion pris en paramètre
    """
    import numpy as np
    from Bwifsttar import time_to_meter_interpolation_mt
    
    func, dist = time_to_meter_interpolation_mt(truck, influence)
    meters = truck.speed * truck.time
    meters = meters[meters<=dist]
    return func(meters)

def reconstruction_norm2(truck, influence_bundle):
    """
        Données : 
            - truck : type namedTuple Truck
            - influence_bundle : influence donnée
        Sortie :
            - predicted : signal reconstruit
    """
    import numpy as np
    from Previous_files.bwim import create_toeplitz

    toeplitz  = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
    T_matrix  = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted

def show_calibration_norm2(trucks, influence):
    """
        Données :
            - trucks : Liste de namedTuple NewTruck
            - influence : Ligne d'influence
        Sorties :
            Void
        Fonction : Affiche les signaux reconstruits et les signaux réels pour chaque truck de la liste trucks (avec les nouveaux namedTuple normalisés)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from Bwifsttar import reconstruction
    
    recovery = reconstruction_norm2(truck, influence)
    print("Meters shape : ",truck.meters.shape)
    print("Recovery shape : ",recovery.shape)
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    delta = truck.meters[1] - truck.meters[0]
    domain = delta*(np.arange(influence.size) - influence.size//2)
    plt.plot(domain, influence)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.xlabel('Meters')
    plt.subplot(1,2,2)
    plt.plot(truck.meters, recovery, linewidth=2, label='Recovered')
    plt.plot(truck.meters, truck.signals.T, label='Observed', alpha=0.7)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.xlabel('Meters')
    plt.legend()
    plt.show()