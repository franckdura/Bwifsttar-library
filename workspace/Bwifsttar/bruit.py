def get_idx_five_meters(meters,distance):
    """
        Données :
            - meters : Liste des mètres pour un camion (truck.time*truck.speed)
            - distance : Distance de laquelle nous voulons déduire l'indice
        Sorties :
            - i : indice de la liste 'meters' (paramètre) à partir duquel le camion a parcouru plus de **distance** mètres
        Fonction : Retourne l'indice à partir duquel on a parcouru plus de **distance** m dans la liste meters.
    """
    i=0

    for meter in meters:
        if meter>distance:
            return i
        
        else:
            i=i+1

def get_std(truck):
    """
        Données :
            -truck : Camion (type namedTuple truck)
        Sorties :
            - std_noise : Ecart-type du bruit du signal d'un camion
        Fonction : Retourne l'écart type du bruit du signal d'un camion par régression linéaire
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import math
    
    meters = truck.speed*truck.time
    idx_five_meters = get_idx_five_meters(meters,5)#on fixe la distance à 5m car c'est là que nous observons en partie le bruit
    x = meters[0:idx_five_meters].reshape(-1,1)
    y = truck.signals[0:idx_five_meters]

    reg = LinearRegression().fit(x, y)
    
    rss = np.sum((y-reg.predict(x))**2)
    std_noise = math.sqrt(rss)
    return std_noise
