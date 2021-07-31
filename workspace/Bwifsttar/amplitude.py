def calibration_mt_amp(trucks, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    """
        Données :
            - trucks : liste des camions de calibration
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation Totale Variance
        Sorties :
            - influence_finale.x : Ligne d'influence estimée à partir des camions de calibration
        Fonction : Calcul la ligne d'influence à partir d'une liste de camions avec une pondération sur le facteur d'amplitude
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

    def func_finale_to_min(h_alpha,T_tilde,y_tilde,ws):
        
        """
            Données : 
                - halpha : la ligne d'influence (que nous cherchons par la minimisation) associée aux coeffs
                de pondération d'amplitude
                - T_tilde : Liste des T
                - y_tilde : Liste des y
                - w : Liste des coeffs de pondération par niveau de bruit du signal
            Sorties :
                - somme_array : Somme des moindres carrés à minimiser
            Fonction :
                Prend en paramètre la LI et les alphas et retourne la somme à minimiser
        """
 
        h = h_alpha[:701]
        alpha = h_alpha[701:]
        alpha_tot = np.ones(0)
        alpha_tot = np.append(alpha_tot,alpha)

        somme_array = 0

        for i in range(T_tilde.shape[0]):

            a = ws[i]*np.linalg.norm(T_tilde[i]@h-(alpha[i])*y_tilde[i])**2

            somme_array += a

        return somme_array
    
    alpha_0 = np.ones(y_tilde.shape[0]-0)
    params_0 = np.concatenate((infl0,alpha_0))

    influence_finale = minimize(func_finale_to_min,params_0,method='SLSQP',args=(T_tilde,y_tilde,ws),tol=0.001)#utiliser CG pour plus rapidité
    print("Alphas finaux : ",influence_finale.x[701:])
    return influence_finale.x[:701]

def lignes_influence_mt_amp(capteurs,nbre_camions,l2_reg=None, tv_reg=None):
    
    """
        Données : 
            - capteurs : chiffre appartenant à [3,4,6,7] identifiant un capteur
            - nbre_camions : nombre de camions sur lesquels faire les calculs de la LI
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation TV
        Sorties :
            - Hc : Liste des lignes d'influence par capteur
        Fonction : Retourne la liste des lignes d'influence par capteur avec régularisation et facteurs d'amplitudes
    
    """
    import numpy as np
    from Bwifsttar import load_senlis_modified,calibration_mt_amp
    
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
        
    for i,capteur in enumerate(capteurs):
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        
        if(nbre_camions<=len(calibration_trucks)):
            
            calibration_trucks = calibration_trucks[0:nbre_camions]
            h = calibration_mt_amp(calibration_trucks,l2_reg,tv_reg)
            Hc[i] = h
        
        else:
            print("Nombre de camions doit être inférieur ou égal à : ",len(calibration_trucks))
        
        
    return Hc


def multi_viz_amp(Hs,capteurs,l2_reg=None, tv_reg=None):
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
    import matplotlib.pyplot as plt
    for i in range(len(Hs)):
        try:
            print("Capteur : ",capteurs[i])


            calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteurs[i]) #[3,4,6,7]
            print("*****CALIBRATION*****")
            plt.figure()
            show_calibration(calibration_trucks, Hc_6[i])
            plt.figure()
            print("*****TRAFFIC*****")
            show_calibration(traffic_trucks, Hc_6[i])

            print("Echec capteur n° : ",capteurs[i])
            plt.show()
        except:
            
            continue

