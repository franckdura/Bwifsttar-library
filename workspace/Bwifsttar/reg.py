def apodization(length, alpha, amplitude=1):
    """
        Données :
            - length : longueur de la ligne d'influence
            - alpha : paramètre pour la génération d'une fenêtre Tukey
        Sorties :
            - win : 
        Fonction : Crée une fenêtre qui "isole" les bords
    """
    import scipy
    import numpy as np
    
    win = 1 - scipy.signal.tukey(length, alpha)
    win = amplitude * win
    win = np.sqrt(win)
    win = np.diag(win)
    return win

def prepare_regularization(A, b, l2_reg=None, tv_reg=None):
    
    import numpy as np
    
    # assert l2_reg is None or tv_reg is None, "Only one regularization must be selected"
    length = A.shape[1]
    total = A.shape[0]
    if l2_reg is not None: #régularisation l2
        win = apodization(length, l2_reg['cutoff'], l2_reg['strength']*total)
        A = np.concatenate((A, win))
        b = np.concatenate((b, np.zeros(length)))
    if tv_reg is not None:#suppression des oscillations
        win  = apodization(length-1, tv_reg['cutoff'], tv_reg['strength']*total)
        diff = np.diag(np.ones(length)) - np.diag(np.ones(length-1),k=1)
        diff = diff[:-1]
        diff = win @ diff
        A = np.concatenate((A, diff))
        b = np.concatenate((b, np.zeros(length-1)))
    return A, b

def calibration_mt_reg(trucks, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
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
    from Bwifsttar import get_std,prepare_least_squares_mt,prepare_regularization
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
    
    #PRÉPARATION DE LA REGULARISATION
    
    for i in range(T_tilde.shape[0]):
        T_tilde[i], y_tilde[i] = prepare_regularization(T_tilde[i], y_tilde[i], l2_reg, tv_reg)#Aucune régularisation pour le moment

    
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

def lignes_influence_mt_reg(capteurs,nbre_camions,l2_reg=None, tv_reg=None):
    
    """
        Données : 
            - capteurs : chiffre appartenant à [3,4,6,7] identifiant un capteur
            - nbre_camions : nombre de camions sur lesquels faire les calculs de la LI
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation TV
        Sorties :
            - Hc : Liste des lignes d'influence par capteur
        Fonction : Retourne la liste des lignes d'influence par capteur avec régularisation
    
    """
    import numpy as np
    from Bwifsttar import load_senlis_modified,calibration_mt_reg
    
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
        
    for i,capteur in enumerate(capteurs):
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        
        if(nbre_camions<=len(calibration_trucks)):
            
            calibration_trucks = calibration_trucks[0:nbre_camions]
            h = calibration_mt_reg(calibration_trucks,l2_reg,tv_reg)
            Hc[i] = h
        
        else:
            print("Nombre de camions doit être inférieur ou égal à : ",len(calibration_trucks))
        
        
    return Hc

def multi_viz_reg(Hs,capteurs,l2_reg=None, tv_reg=None):
    """
        Données :
            - Hs : Liste des lignes d'influence par capteur
            - capteurs : liste des capteurs sur lesquels chargés les camions
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation TV
        Sorties :
            - Void
        Fonction : Fonction qui affiche pour chaque capteur les reconstructions des signaux sur le nombre
        de camions voulu avec possibilité de régularisation
    """
    from Bwifsttar import load_senlis_modified,show_calibration
    for i in range(len(Hs)):
        try:
            print("Capteur : ",capteurs[i])


            calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteurs[i]) #[3,4,6,7]
            show_calibration(calibration_trucks, Hc_6[i])

            print("Echec capteur n° : ",capteurs[i])
        except:
            
            continue
