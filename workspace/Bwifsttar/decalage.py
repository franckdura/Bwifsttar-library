def reconstruction_peaks(truck, influence_bundle,peaks):
    """
        Données : 
            - truck : type namedTuple Truck
            - influence_bundle : influence donnée
            - peaks : Pic à attribuer au camion
        Sortie :
            - predicted : signal reconstruit
            - influence : ligne d'influence adaptée aux meters du camion pris en paramètre
    """
    import numpy as np
    from Previous_files.bwim import create_toeplitz
    from Bwifsttar import time_to_meter_sampling_mt,estimation_peaks

    influence = time_to_meter_sampling_mt(truck, influence_bundle)
    weights = estimation_peaks(truck,peaks,influence)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, peaks)
    T_matrix  = np.sum(weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence

def estimation_peaks(truck,peaks, influence_bundle):
    from Bwifsttar import time_to_meter_sampling_mt
    from Previous_files.bwim import create_toeplitz
    import numpy as np
    
    influence= time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
    return w

def find_best_position_globale(truck,func):#décaler essieu par essieu
    """
        Données :
            - truck : namedTuple Truck
            - func : fonction générant la LI à partir de la vitesse
        Sorties :
            - decalage : décalage global à appliquer aux peaks du truck pour avoir la meilleure pesée
    """
    import numpy as np
    from Bwifsttar import estimation_peaks
    
    var= 10000
    
    for i in range(-10,10):
        influence = func(truck.speed)
        peaks = truck.peaks+i
        w = estimation_peaks(truck,peaks, influence)
        res = eval_LI_peaks(infl,truck,peaks,plot)

        if(res<var):
            decalage = i
            var = res

    return decalage

def get_combinations(truck,decalage):
    """
        Données :
            - truck :namedTuple Truck
            - decalage : intervalle de décalage de chaque peak
        Sorties :
            - combinations : array des combinaisons possibles selon l'intervalle et les peaks du camion
        Fonction : Retourne toutes les combinaisons de peaks selon un intervalle donnée
    """
    import numpy as np
    import itertools
    
    peaks = truck.peaks# (=[1,1,1,1,1,1])
    values = np.arange(-decalage,decalage+1)
    plist = []
    for peak in peaks:
        plist.append(peak + values)
    plist= np.array(plist)
    combinations = list(itertools.product(*plist))
    combinations = np.array(combinations)
    return combinations

def find_best_position_locale(truck,decalage,func):#décaler essieu par essieu
    """
        Données :
            - truck : namedTuple Truck
            - decalage : intervalle dans lequel créer les combinaisons
            - func : Fonction qui génère la LI en fonction de la vitesse
            
        Sorties :
            - dec : retourne la meilleure combinaison de peaks dans un intervalle autour des peaks initiaux
        Fonction : Trouve la meilleure combinaison de peaks pour un camion donnée sur le critère de la pesée en marche (peut être changé facilement)
          
    """
    from Bwifsttar import get_combinations,estimation_peaks,eval_LI_peaks
    import numpy as np
    
    var= 10000
    comb = get_combinations(truck,decalage)
    for i in comb:
        influence = func(truck.speed)
        peaks = i
        w = estimation_peaks(truck,peaks, influence)
        res = eval_LI_peaks(influence,truck,peaks,plot=False)

        if(res<var):
            decalage = i
            var = res
            #print(i)
            #print(var)
    print("Décalage trouvé : ",decalage)
    return decalage

def eval_LI_peaks(influence_estimee,truck,peaks,plot=False):
    """
        Données
            - influence_estimee : Ligne d'influence estimée
            - truck : namedTuple Truck
            - peaks : Pics à attribuer au camion
            - plot : Option pour afficher le signal reconstruit et le vrai signal
        Sorties :
            - de : Distance euclidienne entre le signal réel et le signal reconstruit avec la LI
        Fonction : Calcul la distance euclidienne entre le signal réel et le signal reconstruit à partir du Truck
        et de la LI
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from Bwifsttar import reconstruction_peaks
    if(len(truck.weights) == len(truck.peaks)):
       
        reconstructed, rescaled = reconstruction_peaks(truck, influence_estimee,peaks)
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

def prepare_least_squares_decalage(truck, length,peaks):
    """
        Données :
            - truck :namedTuple Truck
            - length : longueur de la LI
            - peaks : peaks que nous voulons attribuer au truck
        Sorties : 
            - A : MatriceT dans les anciennes notations (voir calibration multitrucks)
            - b : signal
    """
    
    from Previous_files.bwim import create_toeplitz
    import numpy as np
    
    shape = truck.signals.shape#prend les dimensions du signal du camion (y)
    toeplitz = create_toeplitz(shape[-1], length, peaks) #Da 
    A = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)# A = Somme des wa*Da = T
    b = truck.signals # b = y (le signal)
    if len(shape) == 2:
        A = np.tile(A, reps=(shape[0],1))
        b = np.concatenate(b)
    return A, b # retourne le signal et A (correspondant à T dans le cours)

def find_best_peaks(truck,decalage,func1D):
    from Bwifsttar import get_combinations,eval_LI_peaks
    import numpy as np
    import nevergrad as ng
    
    influence = func1D
    comb = get_combinations(truck,decalage)
    idx = np.argsort(comb[0])
    comb[0]=comb[0][idx]
    #print(comb[0])
    def func_to_min(peaks):
            #print("Eval sur la LI : ",eval_LI(influence, truck,peaks))
            #print("Pekas associés : ",peaks)
            #print("\n")
            return(eval_LI_peaks(influence, truck,peaks))
    param = ng.p.Instrumentation(ng.p.Choice(comb))
    optimizer = ng.optimizers.DoubleFastGADiscreteOnePlusOne(parametrization=param, budget=100)
    res_optimizer = np.array(optimizer.minimize(func_to_min).value[0][0])
    res_min = eval_LI_peaks(influence,truck,truck.peaks)
    if (eval_LI_peaks(influence,truck,res_optimizer) < res_min):
        return res_optimizer
    else:
        return truck.peaks
    
    
def calibration_decalage(truck, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    """
        Données :
            - trucks : liste des camions de calibration
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation Totale Variance
        Sorties :
            - influence_finale.x : Ligne d'influence estimée à partir des camions de calibration
        Fonction : Calcul la ligne d'influence à partir d'une liste de camions en cherchant les meilleurs peaks
    """
    from scipy.optimize import minimize
    from Bwifsttar import prepare_least_squares_decalage,find_best_peaks,get_std,prepare_regularization
    import numpy as np

    As, bs = prepare_least_squares_decalage(truck,701,truck.peaks)#retourne T et y
    
    for a in As:
        a = a.astype(float)
    for b in bs:
        b = b.astype(float)
    ws = np.array([])
    ws=np.append(ws,1/get_std(truck))
    ws = ws/(np.sum(ws)) #on centre autour de 1   
    
    As, bs = prepare_regularization(As, bs, l2_reg, tv_reg)#Aucune régularisation pour le moment
      
    infl,_, _, _ = np.linalg.lstsq(As,bs, rcond=None)  

    peaks = find_best_peaks(truck,2,infl)
    print("Peaks : ",peaks)
    As, bs = prepare_least_squares_decalage(truck,701,peaks)#retourne T et y

    
    #def func_finale_to_min(h_alpha):
    def func_finale_to_min(h,As,Bs,ws):
        


        return ws*np.linalg.norm(As@h-bs)**2



    from scipy.optimize import minimize

    influence_alpha = minimize(func_finale_to_min,infl,method='SLSQP',args=(As,bs,ws),tol=0.001)#utiliser CG pour plus rapidité
    #,constraints=cons
    #res = minimize(func_finale_to_min,params_0)#utiliser CG pour plus rapidité
    resultat = influence_alpha.x
    
    

    
    return influence_alpha.x

def calibration_decalage2(truck,peaks, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    """
        Données :
            - trucks : liste des camions de calibration
            - l2_reg : Régularisation L2
            - tv_reg : Régularisation Totale Variance
        Sorties :
            - influence_finale.x : Ligne d'influence estimée à partir des camions de calibration
        Fonction : Calcul la ligne d'influence à partir d'une liste de camions en cherchant les meilleurs peaks
    """
    from scipy.optimize import minimize
    from Bwifsttar import prepare_least_squares_decalage,find_best_peaks,get_std,prepare_regularization
    import numpy as np

    As, bs = prepare_least_squares_decalage(truck,701,truck.peaks)#retourne T et y
    
    for a in As:
        a = a.astype(float)
    for b in bs:
        b = b.astype(float)
    ws = np.array([])
    ws=np.append(ws,1/get_std(truck))
    ws = ws/(np.sum(ws)) #on centre autour de 1   
    
    As, bs = prepare_regularization(As, bs, l2_reg, tv_reg)#Aucune régularisation pour le moment
      
    infl,_, _, _ = np.linalg.lstsq(As,bs, rcond=None)  

    As, bs = prepare_least_squares_decalage(truck,701,peaks)#retourne T et y

    
    #def func_finale_to_min(h_alpha):
    def func_finale_to_min(h,As,Bs,ws):
        


        return ws*np.linalg.norm(As@h-bs)**2

def find_best_position_signal(truck,pos_dec,peaks0,values):#décaler essieu par essieu
    """
        Données : 
            - truck : namedTuple Truck
            - pos_dec : position où doit être fait le décalage dans la liste peaks
            - peaks0 : Peaks initiaux du camion
            - values : valeurs à tester
        Sorties :
            - decalage : valeur à mettre à pos_dec pour avoir le meilleur résultat
    """
    import numpy as np
    from Bwifsttar import calibration_decalage2,eval_LI_peaks
    
    var= 10000
    peaks = peaks0
    for i in values:

        peaks[pos_dec] = i
        infl = calibration_decalage2(truck,peaks,l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95}) 
        
        
        res = eval_LI_peaks(infl,truck,peaks)
        print(res)
        if(res<var):
            decalage = i
            var = res
            #print(i)
            #print(var)
    print("Décalage trouvé")
    return decalage