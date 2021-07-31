def get_SE(repartition):
    
    """
        ***** Get Sous Ensemble *****
        Retourne les deux sous ensembles aléatoirement choisis
        Repartition /30 camions
        
    """
    from Bwifsttar import load_senlis_modified
    import numpy as np
    
    
    
    print("********** Working on SEs... **********")
    calibration_trucks,traffic_trucks = load_senlis_modified(selected=6) #[3,4,6,7]
    calibration_trucks = [truck for truck in calibration_trucks if(len(truck.weights) == len(truck.peaks))]
    traffic_trucks = [truck for truck in traffic_trucks if(len(truck.weights) == len(truck.peaks))]
    ensemble_trucks = []
    se_LI = []
    se_Poids = []

    for i in range(0,len(calibration_trucks)):
        ensemble_trucks.append(calibration_trucks[i])
    for j in range(0,len(traffic_trucks)):
        ensemble_trucks.append(traffic_trucks[i])
    
    n = len(ensemble_trucks)
    idx= np.arange(0,len(ensemble_trucks))
    np.random.shuffle(idx)
    #print(idx)
    for k in idx[:repartition]:
        se_LI.append(ensemble_trucks[k])
    for k in idx[repartition:]:
        se_Poids.append(ensemble_trucks[k])
    

    return se_LI,se_Poids

def get_LI(se_LI):
    """
        Prend en paramètre le sous ensemble de camions pour interpoler la LI /vitesses
        Retourne la fonction interpolée
    """
    import numpy as np
    from Bwifsttar import calibration_decalage
    from scipy.interpolate import interp1d

    
    print("********** Working on LI... **********")
    list_speeds = []
    list_infl = []
    for i,truck in enumerate(se_LI):
        list_speeds.append(se_LI[i].speed)
        infl = calibration_decalage(se_LI[i],l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95})
        list_infl.append(infl)
        
    func1D   = interp1d(list_speeds, list_infl, fill_value="extrapolate",axis=0)#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)
    return func1D


def get_poids(se_Poids,func1D):
    """
        Prend comme paramètre le sous ensemble de poids et la fonction interp LI/Vitesses
        Retourne la liste des poids estimés pour ces camions
    """
    import numpy as np
    from Bwifsttar import find_best_peaks,estimation_peaks
    
    print("********** Working on weights... **********")
    list_poids = []
    for i,truck in enumerate(se_Poids):

        decalage = find_best_peaks(truck,2,func1D(truck.speed))
        w = estimation_peaks(truck,decalage, func1D(truck.speed))
        list_poids.append(w)
    list_poids = np.array(list_poids)
    return list_poids

def get_erreur_essieu(se2,list_poids_estimes):
    import numpy as np
    
    print("********** Working on errors... **********")
    list_erreurs_essieu = []
    list_erreurs_totales = []
    for i,truck in enumerate(se2):
        poids_reel = truck.weights
        poids_total_reel= np.sum(truck.weights)
        poids_estime = list_poids_estimes[i]
        poids_total_estime = np.sum(list_poids_estimes[i])
        diff = abs(poids_reel-poids_estime)
        diff_totale = abs(poids_total_reel-poids_total_estime)
        list_erreurs_essieu.append(diff)
        list_erreurs_totales.append(diff_totale)
    list_erreurs_essieu = np.array(list_erreurs_essieu)
    list_erreurs_totales = np.array(list_erreurs_totales)
    return list_erreurs_essieu,list_erreurs_totales