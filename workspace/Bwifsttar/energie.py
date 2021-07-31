
def get_energie(signal):

    """
        Données :
            - signal : Signal d'un camion
        Sorties :
            - Énergie estimée
        Fonction : Estime l'énergie à partir d'un signal d'un camion
    """
    import numpy as np
    
    return np.sqrt(np.sum(signal*signal.T))

