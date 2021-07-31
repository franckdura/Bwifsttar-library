def get_speeds(trucks):
    """
        Données :
            - trucks : Liste de namedTuple Truck
        Sorties :
            - speeds : Liste des différentes vitesses des camions pris en paramètre
        Fonction : Retourne la liste des vitesses d'une liste de camions
    """
    import numpy as np
    
    speeds = np.array([])
    for truck in trucks :
        if truck.speed not in speeds:
            speeds = np.append(speeds,truck.speed)
    return speeds

def sort_trucks_speeds(trucks):
    """
        Données : 
            - trucks : Liste de namedTuple Truck
        Sorties :
            - trucks70 : Liste de namedTuple Truck roulant à environ 70km/h
            - trucks80 : Liste de namedTuple Truck roulant à environ 80km/h
            - trucks90 : Liste de namedTuple Truck roulant à environ 90km/h
        Fonction : Trie les camions par vitesse dans trois ensemble
    """

    trucks70 = []
    for i in range(len(trucks)):
        if trucks[i].speed<19:
            trucks70.append(trucks[i])
    trucks80 = []
    for i in range(len(trucks)):
        if trucks[i].speed <22 and trucks[i].speed>19:
            trucks80.append(trucks[i])    
    
    trucks90 = []
    for i in range(len(trucks)):
        if trucks[i].speed >22:
            trucks90.append(trucks[i])          

    
    return trucks70,trucks80,trucks90

def lignes_influence_mt_speed(speed,capteurs, l2_reg=None, tv_reg=None):#spped = 70 or 80 or 90
    
    """
        Données : 
            - speed : vitesse des camions à sélectionner pour calculer la LI
            - capteurs : Liste des capteurs sur lesquels nous voulons faire les calculs
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation Total Variation
        Sorties :
            - Hc : Liste des lignes d'influence calculées (une par capteur mis en paramètre)
        Fonction : Retourne la liste des lignes d'influence par capteur et par vitesse
    
    """
    import numpy as np
    from Bwifsttar import sort_trucks_speeds,calibration_mt_reg,load_senlis_modified
    
    
    # On initialise la liste des LI (une par capteur)
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
        
    for i,capteur in enumerate(capteurs):
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        trucks70,trucks80,trucks90 = sort_trucks_speeds(calibration_trucks)
        
        if speed==70:
            h = calibration_mt_reg(trucks70, l2_reg, tv_reg)
            Hc[i] = h
            
        elif speed==80:
            h = calibration_mt_reg(trucks80, l2_reg, tv_reg)
            Hc[i] = h
            
        elif speed==90:
            h = calibration_mt_reg(trucks90, l2_reg, tv_reg)
            Hc[i] = h
        
        else:
            print("Vitesse indisponible")
   
        
    return Hc

def multi_viz_speed(speed,Hs,capteurs):
    """
        Données :
            - speed : Vitesse des camions utilisés pour les calculs
            - Hs : Liste des lignes d'influence par capteur
            - capteurs : capteurs utilisés pour les calculs
        Sorties : 
            void
        Fonction : Affiche les résultats par vitesse et capteur
    """
    from Bwifsttar import load_senlis_modified,sort_trucks_speeds,show_calibration
    for i in range(len(Hs)):
        try:
            print("Capteur : ",capteurs[i])


            calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteurs[i]) #[3,4,6,7]
            trucks70,trucks80,trucks90 = sort_trucks_speeds(calibration_trucks)
            if speed==70:
                show_calibration(trucks70, Hs[i])
            elif speed==80:
                show_calibration(trucks80, Hs[i])
            elif speed==90:
                show_calibration(trucks90,Hs[i])
            else:
                print("Vitesse indisponible")


        except:
            print("Echec capteur n° : ",capteurs[i])
            continue