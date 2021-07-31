import itertools
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from .utils import create_truck
from .plot import show_signal, show_calibration, compare_weights
from .bwim import estimation, calibration
from scipy.optimize import minimize
import scipy.optimize as so
import nevergrad as ng



def load_senlis(selected=slice(3,None)):
    return extract_data('data/senlis/trafic/', 0, [1,2], selected, senlis_weights)


def extract_data(root_path, time_idx, speed_idx, signal_idx, static_weights=None):
    from pathlib import Path
    files = Path(root_path).glob('*.txt')
    trucks = []
    for name in files:
        events  = read_file(name)
        name    = name.stem
        weights = static_weights[name] if static_weights is not None else None
        truck   = create_truck(name, events, time_idx, speed_idx, signal_idx, weights)
        trucks.append(truck)
    return trucks


def read_file(name):
    data = []
    with open(name, 'r') as f:
        for line in f:
            data.append(line.split())
    data = np.array(data, dtype=float).T  # matrix of shape (n_signals, n_samples)
    return data

senlis_weights_modified = {
    ####Trafic####
    
    'PL1_2015-10-01-09-45-52-576':  np.array([7.35, 11.25, 8.20, 8.10, 8.25]),
    'PL2_2015-10-01-09-55-36-371':  np.array([7.65, 14.10, 6.05, 6.20, 6.30]),
    'PL3_2015-10-01-10-22-34-654':  np.array([7.40, 11.60, 8.35, 8.50, 8.50]),
    'PL4_2015-10-01-10-35-43-279':  np.array([7.60, 12.35, 7.80, 7.70, 7.80]),
    'PL5_2015-10-01-10-59-39-060':  np.array([7.35, 10.50, 9.30, 9.70, 9.45]),
    'PL6_2015-10-01-11-05-36-548':  np.array([7.30, 14.30, 6.80, 6.60, 6.70]),
    'PL7_2015-10-01-11-14-06-712':  np.array([7.05, 13.95, 7.25, 7.55, 7.90]),
    'PL8_2015-10-01-11-42-22-222':  np.array([6.40, 14.05, 6.85, 6.75, 6.65]),
    'PL9_2015-10-01-14-42-22-908':  np.array([8.20, 12.00, 8.35, 8.40, 8.35]),
    'PL10_2015-10-01-14-58-50-212': np.array([6.75, 12.45, 8.00, 8.10, 8.00]),
    'PL11_2015-10-01-15-00-57-826': np.array([7.15, 11.40, 8.05, 8.00, 8.45]),
    'PL12_2015-10-01-15-10-30-906': np.array([7.40, 10.80, 8.80, 9.05, 8.75]),
    'PL13_2015-10-01-15-19-48-718': np.array([8.15, 12.20, 7.30, 8.30, 8.80]),
    'PL14_2015-10-01-15-34-03-982': np.array([6.20,  5.05, 7.95, 7.65, 7.40, 7.65]),
    'PL15_2015-10-01-15-36-13-519': np.array([5.35,  8.25, 8.10, 7.75, 7.80, 7.60]),
    
    ####CALIBRATION####
    
    '2015-07-02-10-54-53-222': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-16-36-41-533': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-09-29-10-28-52-687': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-09-56-08-457': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-09-28-46-125': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-15-52-01-990': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-07-02-17-05-11-125': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-16-08-40-480': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-15-40-32-876': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-11-54-10-861': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-07-02-11-24-46-054': np.array([6.03315,  10.8891, 9.17235, 8.58375, 8.77995]), #semi_trailor
    '2015-09-29-15-23-14-921': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-55-32-562': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-28-15-093': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-14-00-38-369': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-12-17-33-320': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-13-33-05-960': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-11-50-31-328': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015
    '2015-09-29-11-08-45-998': np.array([6.16068,  10.8499, 9.08406, 6.98472, 7.0632]), #cal_sep_2015

}

def load_senlis_modified(selected=slice(3,None)):
    traffic_trucks = extract_data('data/senlis/trafic/', 0, [1,2], selected, senlis_weights_modified)
    calibration_trucks = extract_data('data/senlis/calibration/', 0, [1,2], selected, senlis_weights_modified)
    
    return calibration_trucks,traffic_trucks
calibration_trucks,traffic_trucks = load_senlis_modified(selected=6) #[3,4,6,7]

import numpy as np
import scipy.signal
import patsy
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from pathlib import Path
from data import load_senlis
from plot import show_signal, show_calibration, compare_weights

from bwim import create_toeplitz

#mt pour multi trucks
def prepare_least_squares_mt(trucks,length):
    
    trucks_calculables = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    i = 0   
    must_be_normalized = 0
    speed_0= trucks[0].speed
    
    for truck in trucks_calculables :
        if truck.speed != speed_0:
            must_be_normalized = 1
               
    As2= np.empty(len(trucks_calculables),dtype=object) #on initialise bs (signaux des camions de calibration) et As (concaténation des matrices T = somme des wa.Da)
    bs= np.empty(len(trucks_calculables),dtype=object) 

            
    if must_be_normalized:
        
        for truck in trucks_calculables:
            
            signals = truck.signals
            shape = signals.shape#prend les dimensions du signal du camion (y)
            toeplitz = create_toeplitz(shape[-1], length, truck.peaks) #Da 
            
            A = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)# A = Somme des wa*Da = T
            b = signals # b = y (le signal)
            if len(shape) == 2:
                A = np.tile(A, reps=(shape[0],1))
                b = np.concatenate(b)
            

            A = A.astype(float)
            As2[i] = A
            bs[i] = b
            i = i+1

    else:
        
        for truck in trucks_calculables:
            
            signals = truck.signals
            shape = signals.shape#prend les dimensions du signal du camion (y)
            toeplitz = create_toeplitz(shape[-1], length, truck.peaks) #Da 
            
            A = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)# A = Somme des wa*Da = T
            b = signals # b = y (le signal)
            if len(shape) == 2:
                A = np.tile(A, reps=(shape[0],1))
                b = np.concatenate(b)
            

            A = A.astype(float)
            As2[i] = A
            bs[i] = b
            i = i+1
                
    return As2, bs # retourne le signal et A (correspondants à y et T dans les notebooks)

#Ne marche pas s'il n'y a pas le même nombre de peak que de poids
from data import load_senlis
from plot import show_signal, show_calibration, compare_weights
from sklearn.linear_model import LinearRegression
import math

def get_idx_five_meters(meters):
    i=0

    for meter in meters:
        if meter>5:
            return i
        
        else:
            i=i+1

def get_std(truck):
    """retourne l'écart type du bruit du signal d'un camion"""
    meters = truck.speed*truck.time
    idx_five_meters = get_idx_five_meters(meters)
    x = meters[0:idx_five_meters].reshape(-1,1)
    y = truck.signals[0:idx_five_meters]
    reg = LinearRegression().fit(x, y)
    rss = np.sum((y-reg.predict(x))**2)
    return math.sqrt(rss)

def time_to_meter_interpolation_mt(truck, influence): 
    meters = truck.speed * truck.time[:len(influence)]#metres parcourus sur le temps len(influence)
    dist   = meters.max()#le nombre de metres total
    func   = interp1d(meters, influence, fill_value="extrapolate")#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)
    return func, dist#retourne la distance totale et la fonction d'interpolation

def apodization(length, alpha, amplitude=1):
    win = 1 - scipy.signal.tukey(length, alpha)
    win = amplitude * win
    win = np.sqrt(win)
    win = np.diag(win)
    return win

def prepare_regularization(A, b, l2_reg=None, tv_reg=None):
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
        diff = win @ diff#dérivée première
        #diff = win @ diff
        
        A = np.concatenate((A, diff))
        b = np.concatenate((b, np.zeros(length-1)))
    return A, b
 #on va utiliser la même fonction de régularisation qu'avant mais pour chaque matrice du tableau
    
from scipy.optimize import minimize
from scipy.spatial import distance

def calibration_mt(trucks, l2_reg=None, tv_reg=None):# pour le moment aucune régularization
    from scipy.optimize import minimize

    As, bs = prepare_least_squares_mt(trucks,701)#retourne T et y
    for a in As:
        a = a.astype(float)
    for b in bs:
        b = b.astype(float)
    trucks_calculables = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    ws = np.array([])
    for truck in trucks_calculables:
        ws=np.append(ws,1/get_std(truck))
    ws = ws/(np.sum(ws)) #on centre autour de 1   
    
    for i in range(As.shape[0]):
        As[i], bs[i] = prepare_regularization(As[i], bs[i], l2_reg, tv_reg)#Aucune régularisation pour le moment
      
    infl,_, _, _ = np.linalg.lstsq(As[0],bs[0], rcond=None)  

    def func_to_min(h,i):
        return np.linalg.norm(As[i]*h-np.expand_dims(bs[i],axis=1))**2
    
    def sigmoid_arrangee(x):
        return (1/(1+np.exp(x)))+0.8
    
    def contrainte(x):
        if(x<0.3):
            return 500000
        elif(x>1.5):
            return 500000
        else:
            return x
        
    
    #def func_finale_to_min(h_alpha):
    def func_finale_to_min(h,As,bs,ws):
        

        somme_array = 0
        #bs = alpha_tot*bs
        
        #print("BS : ",bs.shape)

        for i in range(As.shape[0]):

            a = ws[i]*np.linalg.norm(As[i]@h-bs[i])**2

            somme_array += a

        return somme_array
    
    alpha_0 = np.ones(bs.shape[0]-0)
    #params_0 = np.array([np.expand_dims(infl,axis=1),np.expand_dims(alpha_0,axis=1)])
    params_0 = np.concatenate((infl,alpha_0))
    #params_0 = np.array([np.expand_dims(infl,axis=1),alpha_0])
    #params_0 = [np.expand_dims(infl,axis=0),np.expand_dims(alpha_0,axis=1)]

    #print("param 0 [1]",params_0[1].shape)
    #print("param 0 [0]",params_0[0].shape)
    #alpha_0 = np.array([1,1,1,1,1,1,1,1,1,1,1])
    #params_0 = infl,alpha_0
    #print("Shape param 0 : ",params_0.shape)
    
    def con_positif(alphas):
        """Somme1 =somme2 pour s'assurer qu'il n'y ait aucune valeur négative"""
        alphas = alphas[701:]
        somme1 = 0
        somme2 =0
        for a in alphas:
            somme1 += a#somme des valeurs
            somme2 += math.sqrt(a**2)#somme des valeurs absolues
        return somme1-somme2
                
    cons = {'type':'eq','fun':con_positif}    

    from scipy.optimize import minimize

    influence_alpha = so.minimize(func_finale_to_min,infl,method='SLSQP',args=(As,bs,ws),tol=0.001)#utiliser CG pour plus rapidité
    #,constraints=cons
    #res = minimize(func_finale_to_min,params_0)#utiliser CG pour plus rapidité

    return influence_alpha.x


def time_to_meter_sampling_mt(truck, influence):
    func, dist = time_to_meter_interpolation_mt(truck, influence)
    meters = truck.speed * truck.time
    meters = meters[meters<=dist]
    return func(meters)#donne les valeurs via fonction d'interpolation des 'influence' (temps) pour une distance maximum

def reconstruction(truck, influence_bundle):
    influence = time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, truck.peaks)
    T_matrix  = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence

def show_calibration(trucks, influence):
    trucks_calculables = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    i=0

    fig, axs = plt.subplots(3, 4,figsize=(17, 6))
    axs = axs.ravel()
    
    for truck in trucks_calculables:   
        reconstructed, rescaled = reconstruction(truck, influence)
        meters = truck.speed * truck.time
        plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
        plt.xlabel('Meters')
        axs[i].plot(meters, reconstructed, linewidth=2, label='Recon.')
        axs[i].plot(meters, truck.signals.T, label='Observed', alpha=0.7)
        axs[i].set_title(str("Meters"))
        i=i+1
    plt.figure()
    plt.plot(meters[:len(rescaled)], rescaled)
    plt.xlabel("Meters")
    plt.title("Ligne d'nfluence")
    plt.show()
from plot import show_signal
from bwim import estimation, calibration

#show_calibration(calibration_trucks, inf)


def lignes_influence_mt(capteurs,nbre_camions, l2_reg=None, tv_reg=None):
    
    """
        Retourne la liste des lignes d'influence par capteur
    
    """
    
    
    
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
    
    i=0
    
    for capteur in capteurs:
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        
        if(nbre_camions<=len(calibration_trucks)):
            
        
            calibration_trucks = calibration_trucks[0:nbre_camions]
            h = calibration_mt(calibration_trucks, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1
        
        else:
            print("Nombre de camions doit être inférieur ou égal à : ",len(calibration_trucks))
        
        
    return Hc

def multi_viz(Hs,capteurs):
    for i in range(len(Hs)):
        try:
            print("Capteur : ",capteurs[i])
            

            calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteurs[i]) #[3,4,6,7]
            show_calibration(calibration_trucks, Hc_6[i])

        except:
            print("Echec capteur n° : ",capteurs[i])
            continue

def estimation(truck, influence_bundle):
    influence= time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
    return w
def estimation_mt(trucks, influence_bundle):
    list_w = []
    trucks = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    for truck in trucks:
        
        influence= time_to_meter_sampling_mt(truck, influence_bundle)
        toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
        H_matrix = toeplitz @ influence
        w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
        list_w.append(w)
    return list_w
def compare_weights_mt(list_estimated, trucks):
    i=0
    errors = np.array([])
    trucks = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    for estimated in list_estimated:
        
        index = np.arange(len(estimated)) + 1
        bar_width = 0.45
        ver_shift = 1
        opacity = 0.8
        error = error = np.abs(estimated - trucks[i].weights).sum()
        print("Vitesse du camion ci dessous en km/h : ",trucks[i].speed*3.6)
        plt.figure(figsize=(9,5))
        plt.bar(index-bar_width/2, estimated,    bar_width, alpha=opacity, color='b', label='Pesées en marche')
        plt.bar(index+bar_width/2, trucks[i].weights, bar_width, alpha=opacity, color='r', label='Pesées statiques')
        plt.title('Erreur total: {:2.2f} t'.format(error), fontsize=14)
        plt.xlabel('Essieu', fontsize=14)
        plt.ylabel('Poids', fontsize=14)
        plt.legend(fontsize=14)
        for x, y, z in zip(index, trucks[i].weights, estimated):
            plt.text(x+bar_width/2, y-ver_shift, '%.2f' % y, fontsize=12, fontweight='bold', color='white', ha='center')
            plt.text(x-bar_width/2, z-ver_shift, '%.2f' % z, fontsize=12, fontweight='bold', color='white', ha='center')
        plt.show()
        i=i+1
        errors = np.append(errors,error)
    return errors


def sort_trucks_speeds(trucks):
    idx_70 = [0,2,5,7]
    idx_90 = [1,4,8,9,10,11]
    idx_80 = [3,6]
    trucks70 = []
    for i in range(len(trucks)):
        if i in idx_70:
            trucks70.append(trucks[i])
    trucks80 = []
    for i in range(len(trucks)):
        if i in idx_80:
            trucks80.append(trucks[i])    
    
    trucks90 = []
    for i in range(len(trucks)):
        if i in idx_90:
            trucks90.append(trucks[i])          

    
    return trucks70,trucks80,trucks90



def lignes_influence_mt_speed(speed,capteurs, l2_reg=None, tv_reg=None):#spped = 70 or 80 or 90
    
    """
        Retourne la liste des lignes d'influence par capteur et par vitesse
    
    """
    
    
    
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
    
    i=0
    
    for capteur in capteurs:
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        trucks70,trucks80,trucks90 = sort_trucks_speeds(calibration_trucks)
        if speed==70:

            h = calibration_mt(trucks70, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1
        elif speed==80:
            h = calibration_mt(trucks80, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1      
        elif speed==90:
            h = calibration_mt(trucks90, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1     
        
        else:
            print("Vitesse indisponible")
   
        
    return Hc

def find_best_position(truck):#décaler essieu par essieu
    var= 10000
    
    for i in range(-10,10):
        influence = func1D(truck.speed)
        peaks = truck.peaks+i
        w = estimation_peaks(truck,peaks, influence)
        #print("W : ",w)
        poids_mesure = np.sum(w)
        #print(poids_mesure)
        poids_reel = np.sum(truck.weights)
        #print(poids_reel)
        #print("Diff : ",np.sum(abs(truck.weights -w)))
        if(np.sum(abs(truck.weights -w))<var):
            decalage = i
            var = np.sum(abs(truck.weights -w))
            #print(i)
            #print(var)
    return decalage
def estimation_peaks(truck,peaks, influence_bundle):
    influence= time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
    return w


def get_combinations(truck,decalage):
    peaks = truck.peaks# (=[1,1,1,1,1,1])
    values = np.arange(-decalage,decalage+1)
    plist = []
    for peak in peaks:
        plist.append(peak + values)
    plist= np.array(plist)
    combinations = list(itertools.product(*plist))
    combinations = np.array(combinations)
    return combinations

def reconstruction_peaks(truck, influence_bundle,peaks):
    influence = time_to_meter_sampling_mt(truck, influence_bundle)
    toeplitz  = create_toeplitz(truck.time.size, influence.size, peaks)
    T_matrix  = np.sum(truck.weights[:,None,None] * toeplitz, axis=0)
    predicted = T_matrix @ influence
    return predicted, influence

def eval_LI(influence_estimee,truck,peaks):
    de=100000000
    if(len(truck.weights) == len(truck.peaks)):
       
        reconstructed, rescaled = reconstruction_peaks(truck, influence_estimee,peaks)
        meters = truck.speed * truck.time
        diff = truck.signals.T - reconstructed
        de = np.linalg.norm(truck.signals.T - reconstructed)
        #print("Distance euclidienne :" ,de)

       
    else:
        
        print("Truck non calculable")
    return de

def find_best_position_signal(truck,decalage,func1D):#décaler essieu par essieu
    var= 10000
    comb = get_combinations(truck,decalage)
    for i in comb:
        influence = func1D(truck.speed)
        peaks = i
        res = eval_LI(influence,truck,peaks)
        if(res<var):
            decalage = i
            var = res
            #print(i)
            #print(var)
    print("Décalage trouvé")
    return decalage

def find_best_peaks(truck,decalage,func1D):
    
    influence = func1D(truck.speed)
    comb = get_combinations(truck,decalage)
    idx = np.argsort(comb[0])
    comb[0]=comb[0][idx]
    #print(comb[0])
    def func_to_min(peaks):
            #print("Eval sur la LI : ",eval_LI(influence, truck,peaks))
            #print("Pekas associés : ",peaks)
            #print("\n")
            return(eval_LI(influence, truck,peaks))
    param = ng.p.Instrumentation(ng.p.Choice(comb))
    optimizer = ng.optimizers.DoubleFastGADiscreteOnePlusOne(parametrization=param, budget=100)
    res_optimizer = np.array(optimizer.minimize(func_to_min).value[0][0])
    res_min = eval_LI(influence,truck,truck.peaks)
    if (eval_LI(influence,truck,res_optimizer) < res_min):
        return res_optimizer
    else:
        return truck.peaks
    

def lignes_influence_mt_speed(speed,capteurs, l2_reg=None, tv_reg=None):#spped = 70 or 80 or 90
    
    """
        Retourne la liste des lignes d'influence par capteur et par vitesse
    
    """
    
    
    
    Hc = np.empty(len(capteurs),dtype=object)#liste des lignes d'influence (autant que de capteurs à tester)
    
    i=0
    
    for capteur in capteurs:
        calibration_trucks,traffic_trucks = load_senlis_modified(selected=capteur) #[3,4,6,7]
        trucks70,trucks80,trucks90 = sort_trucks_speeds(calibration_trucks)
        if speed==70:

            h = calibration_mt(trucks70, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1
        elif speed==80:
            h = calibration_mt(trucks80, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1      
        elif speed==90:
            h = calibration_mt(trucks90, l2_reg, tv_reg)
            Hc[i] = h
            i= i+1     
        
        else:
            print("Vitesse indisponible")
   
        
    return Hc

def multi_viz_speed(speed,Hs,capteurs):
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
 

