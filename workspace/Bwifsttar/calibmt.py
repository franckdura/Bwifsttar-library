def prepare_least_squares_mt(trucks,length):
    """
        Données : 
            - trucks : Liste des namedTuples truck 
            - length : Longueur de la ligne d'influence (classiquement fixée à 701)
        Sorties :
            - As2 : Liste des T (somme Teoplitz*poids) définis ci-dessus (ou T tilde)
            - bs : Liste des signaux (ou y tilde)
        
        Fonction : Retourne T-tilde et y-tilde, servant par la suite pour la calibration, à partir
        de la liste des camions de calibration et la longueur de la ligne d'influence (LI)
    
    """
    import numpy as np
    from Previous_files.bwim import create_toeplitz
    
    trucks_calculables = [truck for truck in trucks if(len(truck.weights)==len(truck.peaks))]#Garde les camions dont les poids ont chacun un peaks sur le signal
    
    
    #on initialise y_tilde (signaux des camions de calibration) et T_tilde (concaténation des matrices T = somme des wa.Da)
    T_tilde = np.empty(len(trucks_calculables),dtype=object) #car nous stockons des matrices de tailles différentes
    y_tilde = np.empty(len(trucks_calculables),dtype=object) 
  
    for i,truck in enumerate(trucks_calculables):

        signals = truck.signals
        shape = signals.shape #prend les dimensions du signal du camion (y)
        toeplitz = create_toeplitz(shape[-1], length, truck.peaks) #Da 

        T = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)# T = Somme des wa*Da 
        y = signals # y (le signal)
        if len(shape) == 2:
            T = np.tile(T, reps=(shape[0],1))
            y = np.concatenate(y)
            
        T = T.astype(float)
        T_tilde[i] = T
        y_tilde[i] = y
        
    return T_tilde, y_tilde # retourne le signal et A (correspondants à y et T dans les notebooks)


def calibration_mt(trucks, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    """
        Données :
            - trucks : liste des camions de calibration
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
    
    T_tilde, y_tilde = prepare_least_squares_mt(trucks,701)
    
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

def poids_tot(truck,influence):
    """
        Données :
            - truck : namedTuple de Truck
            - influence : Ligne d'influence
        Sorties :
            - poids_tot : Poids total estimé
        Fonction : retourne le poids total estimé d'un camion à partir de la ligne d'influence et du signal du camion
    """
    import numpy as np
    
    new_signal= []
    for i in range(truck.peaks.shape[0]):
        new_signal.append(truck.signals[truck.peaks[i]])
    new_influence = []
    for i in range(truck.peaks.shape[0]):
        new_influence.append(influence[truck.peaks[i]])
    #return (np.sum(new_signal)/np.sum(new_influence))
    poids_tot = (np.sum(truck.signals)/np.sum(influence))
    return poids_tot
