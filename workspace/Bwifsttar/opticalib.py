def reconstruction_eval(truck, weights,influence_bundle):
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
    from Bwifsttar import time_to_meter_sampling_mt

    influence = time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, truck.peaks)
    T_matrix  = np.sum(weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence

def eval_LI(influence_estimee,truck,plot=False):
    """
        Données
            - influence_estimee : Ligne d'influence estimée
            - truck : namedTuple Truck
            - plot : Option pour afficher le signal reconstruit et le vrai signal
        Sorties :
            - de : Distance euclidienne entre le signal réel et le signal reconstruit avec la LI
        Fonction : Calcul la distance euclidienne entre le signal réel et le signal reconstruit à partir du Truck
        et de la LI
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from Bwifsttar import reconstruction
    if(len(truck.weights) == len(truck.peaks)):
       
        reconstructed, rescaled = reconstruction(truck, influence_estimee)
        meters = truck.speed * truck.time

        diff = truck.signals.T - reconstructed
 
        de = np.linalg.norm(truck.signals.T - reconstructed)
        if plot:
            plt.figure()
            plt.plot(meters, reconstructed, linewidth=2, label='Recon.')
            plt.plot(meters, truck.signals.T, label='Observed', alpha=0.7)
            plt.xlabel("Distance euclidienne : " + str(de))
            plt.legend()
            plt.show()
    else:
        print("Truck non calculable")
    return de

def eval_LI_mt(trucks,influence,plot=False):
    """
        - Données :
            - trucks : Liste de namedTuple Truck
            - influence : Ligne d'influence calculée
            - plot :  Option permettant d'afficher 
        - Sorties :
            - Distance euclidienne moyenne de tous les camions pris en paramètres
            
    """
    
    import numpy as np
    from Bwifsttar import eval_LI
    
    trucks = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    list_de = []
    for truck in trucks:
        list_de.append(eval_LI(influence,truck,plot))
    return(np.mean(np.array([list_de])))



def time_to_meter_interpolation_opti(truck, influence,linspace = 5000): 
    """
        Données :
            - truck : namedTuple truck
            - influence : LI calculée resamplée
            - linspace : Nombre d'échantillons de la LI resamplée
        Sorties : 
            - func : Fonction interpolée
            - dist : distance max
        Fonction : Interpole la LI selon les meters
    """
    from scipy.interpolate import interp1d
    import numpy as np
    
    meters = np.linspace(0,30,linspace)
    dist   = meters.max()#le nombre de metres total
    func   = interp1d(meters, influence, fill_value="extrapolate")#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)
    return func, dist#retourne la distance totale et la fonction d'interpolation

def time_to_meter_sampling_opti(truck, influence,linspace = 5000):
    """
        Données :
            - truck : namedTuple Truck
            - influence : Ligne d'influence calculée
            - linspace : Nombre d'échantillons de la LI resamplée

        Sorties :
            - LI resamplée sur les meters du camion
        Fonction : Permet de resampler la ligne d'influence en fonction des meters
    """
    
    import numpy as np
    from Bwifsttar import time_to_meter_interpolation_opti
    
    func, dist = time_to_meter_interpolation_opti(truck, influence,linspace)
    meters = truck.speed * truck.time
    meters = meters[meters<=dist]
    return func(meters)#donne les valeurs via fonction d'interpolation des 'influence' (temps) pour une distance maximum


def reconstruction_opti(truck, influence_bundle,linspace = 5000):
    """
        Données : 
            - truck : type namedTuple Truck
            - influence_bundle : influence donnée
            - linspace : Nombre d'échantillons de la LI resamplée
        Sortie :
            - predicted : signal reconstruit
            - influence : ligne d'influence adaptée aux meters du camion pris en paramètre
    """
    import numpy as np
    from Previous_files.bwim import create_toeplitz
    from Bwifsttar import time_to_meter_sampling_opti

    influence = time_to_meter_sampling_opti(truck, influence_bundle,linspace)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, truck.peaks)
    T_matrix  = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence

def eval_LI_opti(influence_estimee,truck,plot=False,linspace = 5000):
    """
        Données
            - influence_estimee : Ligne d'influence estimée
            - truck : namedTuple Truck
            - plot : Option pour afficher le signal reconstruit et le vrai signal
            - linspace : Nombre d'échantillons de la LI resamplée

        Sorties :
            - de : Distance euclidienne entre le signal réel et le signal reconstruit avec la LI
        Fonction : Calcul la distance euclidienne entre le signal réel et le signal reconstruit à partir du Truck
        et de la LI
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from Bwifsttar import reconstruction_opti
    
    if(len(truck.weights) == len(truck.peaks)):
       
        reconstructed, rescaled = reconstruction_opti(truck, influence_estimee,linspace)
        meters = truck.speed * truck.time

        diff = truck.signals.T - reconstructed
 
        de = np.linalg.norm(truck.signals.T - reconstructed)
        if plot:
            plt.figure()
            plt.plot(meters, reconstructed, linewidth=2, label='Recon.')
            plt.plot(meters, truck.signals.T, label='Observed', alpha=0.7)
            plt.xlabel("Distance euclidienne : " + str(de))
            plt.legend()
            plt.show()
        return de
    else:
        print("Truck non calculable")
