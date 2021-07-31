def time_to_meter_interpolation_mt(truck, influence): 
    
    """
        Données : 
            - truck : camion (type namedTuple Truck)
            - influence : Ligne d'influence
        Sorties :
            - func : fonction qui résulte de l'interpolation de la ligne d'influence avec les metres (time.time * time.speed)
            - dist : distance maximale du vecteur meters
        Fonction : Retourne la fonction interpolée de la LI avec les meters du camion et retourne la distance max.
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    meters = truck.speed * truck.time[:len(influence)]#metres parcourus sur le temps len(influence)
    dist   = meters.max()#le nombre de metres total
    func   = interp1d(meters, influence, fill_value="extrapolate")#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)

    return func, dist#retourne la distance totale et la fonction d'interpolation

def time_to_meter_sampling_mt(truck, influence):
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
    return func(meters)#donne les valeurs via fonction d'interpolation des 'influence' (temps) pour une distance maximum


def reconstruction(truck, influence_bundle):
    """
        Données : 
            - truck : type namedTuple Truck
            - influence_bundle : influence donnée
        Sortie :
            - predicted : signal reconstruit
            - influence : ligne d'influence adaptée aux meters du camion pris en paramètre
    """
    import numpy as np
    from Previous_files.bwim import create_toeplitz
    from Bwifsttar import time_to_meter_sampling_mt,estimation

    influence = time_to_meter_sampling_mt(truck, influence_bundle)
    weights = estimation(truck,influence)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, truck.peaks)
    T_matrix  = np.sum(weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence



def show_calibration(trucks, influence):
    """
        Données :
            - trucks : Liste de namedTuple Truck
            - influence : Ligne d'influence
        Sorties :
            Void
        Fonction : Affiche les signaux reconstruits et les signaux réels pour chaque truck de la liste trucks
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    trucks_calculables = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]

    fig, axs = plt.subplots(4, 4,figsize=(17, 6))
    axs = axs.ravel()
    
    for i,truck in enumerate(trucks_calculables):   
        reconstructed, rescaled = reconstruction(truck, influence)
        meters = truck.speed * truck.time
        plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
        plt.xlabel('Meters')
        axs[i].plot(meters, reconstructed, linewidth=2, label='Recon.')
        axs[i].plot(meters, truck.signals.T, label='Observed', alpha=0.7)
        axs[i].set_title(str("Meters"))

    plt.figure()
    plt.plot(meters[:len(rescaled)], rescaled)
    plt.xlabel("Meters")
    plt.title("Ligne d'nfluence")
    plt.show() 

    
def lignes_influence_mt(capteurs,nbre_camions):
    
    """
        Données : 
            - capteurs : chiffre appartenant à [3,4,6,7] identifiant un capteur
            - nbre_camions : nombre de camions sur lesquels faire les calculs de la LI
        Sorties :
            - Hc : Liste des lignes d'influence par capteur
        Fonction : Retourne la liste des lignes d'influence par capteur
    
    """
    import numpy as np
    from Bwifsttar import load_senlis_modified
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
        
    for i,capteur in enumerate(capteurs):
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        
        if(nbre_camions<=len(calibration_trucks)):
            
            calibration_trucks = calibration_trucks[0:nbre_camions]
            h = calibration_mt(calibration_trucks)
            Hc[i] = h
        
        else:
            print("Nombre de camions doit être inférieur ou égal à : ",len(calibration_trucks))
        
        
    return Hc
       
    
def multi_viz(Hs,capteurs):
    """
        Données :
            - Hs : Liste des lignes d'influence par capteur
            - capteurs : liste des capteurs sur lesquels chargés les camions
        Sorties :
            - Void
        Fonction : Fonction qui affiche pour chaque capteur les reconstructions des signaux sur le nombre
        de camions voulu
    """
    from Bwifsttar import load_senlis_modified

    for i in range(len(Hs)):
        try:
            print("Capteur : ",capteurs[i])
            

            calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteurs[i]) #[3,4,6,7]
            show_calibration(calibration_trucks, Hc_6[i])

        except:
            print("Echec capteur n° : ",capteurs[i])
            continue
