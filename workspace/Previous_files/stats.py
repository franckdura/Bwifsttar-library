import itertools
import numpy as np
from .interpolation_LI import find_best_peaks
from .interpolation_LI import get_combinations
from .interpolation_LI import estimation_peaks
from .interpolation_LI import find_best_position_signal
from .interpolation_LI import reconstruction_peaks
from .interpolation_LI import eval_LI
from scipy.interpolate import interp1d
import scipy.optimize as so
from scipy.optimize import minimize
from .interpolation_LI import time_to_meter_interpolation_mt,prepare_least_squares_mt,get_std,prepare_regularization,estimation
from .interpolation_LI import calibration_mt
from .interpolation_LI import load_senlis_modified
import random
import matplotlib.pyplot as plt

def get_SE3(repartition):
    #repartition en %
    pcalib_LI=1.8
    ptraffic_LI = 0.2

    repartition = 0.01*repartition
    calibration_trucks,traffic_trucks = load_senlis_modified(selected=6) #[3,4,6,7]
    calibration_trucks = [truck for truck in calibration_trucks if(len(truck.weights) == len(truck.peaks))]
    traffic_trucks = [truck for truck in traffic_trucks if(len(truck.weights) == len(truck.peaks))]
    se_LI = []
    se_Poids = []
    n_calib_LI = round(repartition*len(calibration_trucks)*pcalib_LI)
    n_traffic_LI = round(repartition*len(traffic_trucks)*ptraffic_LI)

    idx_calib_LI = np.arange(0,len(calibration_trucks))
    np.random.shuffle(idx_calib_LI)
    idx_traffic_LI = np.arange(0,len(traffic_trucks))
    np.random.shuffle(idx_traffic_LI)

    trucks_calib_LI = []
    for i in range(n_calib_LI):
        trucks_calib_LI.append(calibration_trucks[idx_calib_LI[i]])

    trucks_traffic_LI = []
    for i in range(n_traffic_LI):
        trucks_traffic_LI.append(traffic_trucks[idx_traffic_LI[i]])

    trucks_calib_Poids = []
    for i,truck in enumerate(calibration_trucks):
        if truck not in trucks_calib_LI:
            trucks_calib_Poids.append(calibration_trucks[i])

    trucks_traffic_Poids = []
    for i,truck in enumerate(traffic_trucks):
        if truck not in trucks_traffic_LI:
            trucks_traffic_Poids.append(traffic_trucks[i])

    for i in range(0,len(trucks_calib_LI)):
        se_LI.append(trucks_calib_LI[i])

    for i in range(0,len(trucks_traffic_LI)):
        se_LI.append(trucks_traffic_LI[i])

    for j in range(0,len(trucks_traffic_Poids)):
        se_Poids.append(trucks_traffic_Poids[j])

    for j in range(0,len(trucks_calib_Poids)):
        se_Poids.append(trucks_calib_Poids[j])


    return se_LI,se_Poids

def get_LI(se_LI):
    """
        Prend en paramètre le sous ensemble de camions pour interpoler la LI /vitesses
        Retourne la fonction interpolée
    """
    print("********** Working on LI... **********")
    list_speeds = []
    list_infl = []
    for i,truck in enumerate(se_LI):
        list_speeds.append(se_LI[i].speed)
        infl = calibration_mt(se_LI[i:i+1],l2_reg={'strength': 1e3, 'cutoff': 0.01},tv_reg={'strength': 1e3, 'cutoff': 0.95})
        list_infl.append(infl)

    func1D   = interp1d(list_speeds, list_infl, fill_value="extrapolate",axis=0)#permet à partir de meters et infuence de trouver une approximation de influence = f(meters)
    return func1D

def get_poids(se_Poids,func1D):
    """

        Prend comme paramètre le sous ensemble de poids et la fonction interp LI/Vitesses
        Retourne la liste des poids estimés pour ces camions
    """
    print("********** Working on weights... **********")
    list_poids = []
    for i,truck in enumerate(se_Poids):
        #print(str(i+1)+"/"+str(len(se_Poids)))
        #decalage  = find_best_position_signal(truck,2,func1D)
        decalage = find_best_peaks(truck,2,func1D)
        #decalage = truck.peaks
        w = estimation_peaks(truck,decalage, func1D(truck.speed))
        list_poids.append(w)
    list_poids = np.array(list_poids)
    return list_poids


def get_erreur_essieu(se2,list_poids_estimes):
    print("********** Working on errors... **********")
    list_erreurs_essieu = []
    list_erreurs_totales = []
    for i,truck in enumerate(se2):
        poids_reel = truck.weights
        poids_total_reel= np.sum(truck.weights)
        poids_estime = list_poids_estimes[i]
        poids_total_estime = np.sum(list_poids_estimes[i])
        diff = abs(poids_reel-poids_estime)
        diff_totale = abs(poids_total_reel-poids_total_estime)
        list_erreurs_essieu.append(diff)
        list_erreurs_totales.append(diff_totale)
    list_erreurs_essieu = np.array(list_erreurs_essieu)
    list_erreurs_totales = np.array(list_erreurs_totales)
    return list_erreurs_essieu,list_erreurs_totales


def get_metrics(list_erreurs_essieu,list_erreurs_totales,se2):
    print("**********  Working on metrics... **********")
    """
        Prend en params les erreurs sur essieux et les erreurs totales

        Retourne :
        - l'erreur total moyenne sur l'ensemble des camions de test, ok
        - la liste des % d'erreur sur le poids total de chaque camion de test ok
        - le % d'erreur moyen sur les poids totaux des camions de test, ok
        - la liste des erreurs moyennes par essieu, ok
        - la liste des % d'erreur moyen par essieu, ok
    """
    poids_totaux = []
    poids = []
    for truck in se2:
        poids_totaux.append(np.sum(truck.weights))
        poids.append(truck.weights)
    poids = np.array(poids)
    poids_essieu_moyen = []
    for poid in poids.T:
        poids_essieu_moyen.append(np.sum(poid)/len(poid))


    scores_moyen_essieux = []
    percent_moyen_essieux = []
    score_moyen_total = np.sum(list_erreurs_totales)/len(list_erreurs_totales)
    percent_erreur_totale = []

    for j,i in enumerate(list_erreurs_essieu.T):
        scores_moyen_essieux.append(np.sum(i)/len(i))

    for j,i in enumerate(poids_essieu_moyen):
        val = 1 - (poids_essieu_moyen[j]-scores_moyen_essieux[j])/poids_essieu_moyen[j]
        percent_moyen_essieux.append(val)

    for j,i in enumerate(poids_totaux):
        val = 1 - (poids_totaux[j]-list_erreurs_totales[j])/poids_totaux[j]
        percent_erreur_totale.append(val)

    percent_erreur_totale_moy = np.sum(percent_erreur_totale)/len(percent_erreur_totale)

    return np.array(score_moyen_total),np.array([percent_erreur_totale])*100,np.array([percent_erreur_totale_moy])*100,np.array(scores_moyen_essieux),100*np.array(percent_moyen_essieux)
import matplotlib.pyplot as plt
import collections

def get_stats3(iterations,repartition):
    """
        Prend en params le nombre d'itérations voulues et la repartition dans les sous ensemble
        Retourne :
            - L'erreur totale moyenne (t)
            - L'erreur moyenne par essieu (t)
            - Le % d'erreur moyen total (%)
            - Le % d'erreur moyen total par essieu (%)
        Affiche :
            - Le diagramme baton des erreurs sur les poids totaux
            - Le diagrame baton des erreurs sur les poids totaux redressés
            - Le diagramme baton des erreurs par essieu
    """
    list_erreur_tot_moy = []
    list_erreur_essieu_moy = np.zeros(5)
    list_percent_erreur_moy_tot = []
    list_percent_erreur_essieu = np.zeros(5)
    list_erreurs_tot = []
    list_erreurs_tot_redr = []
    list_erreur_essieu1 = []
    list_erreur_essieu2 = []
    list_erreur_essieu3 = []
    list_erreur_essieu4 = []
    list_erreur_essieu5 = []

    list_percent_tot = []
    for i in range(iterations):
        try:
            print("Itération n° : "+str(i+1)+"/"+str(iterations))
            se2,se1 = get_SE3(repartition)
            print("LEN SE1 : ",len(se1))
            print("LEN SE2 : ",len(se2))
            func = get_LI(se1)
            poids_estimes = get_poids(se2,func)
            scores,scores_totaux = get_erreur_essieu(se2,poids_estimes)
            erreur_tot_moy,list_percent_err_poids_tot,percent_err_moy,list_err_moy_essieux,list_percent_moy_essieux = get_metrics(scores,scores_totaux,se2)
            for a,j in enumerate(poids_estimes):
                list_erreurs_tot.append(np.sum(j)-np.sum(se2[a].weights))
                list_erreurs_tot_redr.append(abs(np.sum(j)-np.sum(se2[a].weights)))

                list_percent_tot.append(100*(list_erreurs_tot[a]/np.sum(se2[a].weights)))
                #print(1-100*(list_erreurs_tot[a]/np.sum(se2[a].weights)))
                for a,j in enumerate(poids_estimes.T[0]):
                    list_erreur_essieu1.append(j-se2[a].weights[0])
                for a,j in enumerate(poids_estimes.T[1]):
                    list_erreur_essieu2.append(j-se2[a].weights[1])
                for a,j in enumerate(poids_estimes.T[2]):
                    list_erreur_essieu3.append(j-se2[a].weights[2])
                for a,j in enumerate(poids_estimes.T[3]):
                    list_erreur_essieu4.append(j-se2[a].weights[3])
                for a,j in enumerate(poids_estimes.T[4]):
                    list_erreur_essieu5.append(j-se2[a].weights[4])
            #print("\n\n\n\nMetrics sur les poids totaux : \n\n")
            #print("Liste des erreurs totales (t) : ",scores_totaux)
            #print("\nErreur totale moyenne sur l'ensemble des camions de test (t) : ",erreur_tot_moy)
            list_erreur_tot_moy.append(erreur_tot_moy)
            #print("\nListe des % d'erreur sur le poids total de chaque camion de test (%)",list_percent_err_poids_tot)
            #print("\n% d'erreur moyen sur les poids totaux des camions de test (%)",percent_err_moy)
            list_percent_erreur_moy_tot.append(percent_err_moy)
            #print("\n\nMetrics sur les poids par essieu : \n\n")
            #print("Scores par essieu (t) \n", scores)
            #print("\nListe des erreurs moyennes par essieu (t) : ",list_err_moy_essieux)
            list_erreur_essieu_moy = np.add(list_erreur_essieu_moy,list_err_moy_essieux)
            #print("\nListe des % d'erreur moyen par essieu (%) : ",list_percent_moy_essieux)
            list_percent_erreur_essieu = np.add(list_percent_erreur_essieu,list_percent_moy_essieux)
        except:
            continue

    erreur_tot_moy = np.sum(np.array(list_erreur_tot_moy)/len(list_erreur_tot_moy))
    erreur_essieu_moy =list_erreur_essieu_moy/len(list_erreur_essieu_moy)
    percent_err_tot_moy = np.sum(list_percent_erreur_moy_tot)/len(list_percent_erreur_moy_tot)
    percent_erreur_essieux = list_percent_erreur_essieu/len(list_percent_erreur_essieu)
    list_erreurs_tot=np.array(list_erreurs_tot)
    list_erreurs_tot_redr=np.array(list_erreurs_tot_redr)
    list_erreur_essieu1=np.array(list_erreur_essieu1)
    list_erreur_essieu2=np.array(list_erreur_essieu2)
    list_erreur_essieu3=np.array(list_erreur_essieu3)
    list_erreur_essieu4=np.array(list_erreur_essieu4)
    list_erreur_essieu5=np.array(list_erreur_essieu5)


    return list_erreurs_tot,list_erreurs_tot_redr,list_percent_tot,percent_err_tot_moy,list_erreur_essieu1,list_erreur_essieu2,list_erreur_essieu3,list_erreur_essieu4,list_erreur_essieu5

if __name__ == '__main__':
    let,letr,lpt,lpt,lee1,lee2,lee3,lee4,lee5= get_stats3(10,20)
    

    plt.figure()
    plt.hist(let)
    plt.title("Erreurs poids totaux (t)")
    plt.show()

    plt.figure()
    plt.hist(letr)
    plt.title("Erreurs poids totaux redressés (t)")
    plt.show()


    plt.figure()
    plt.hist(lpt)
    plt.title("Erreurs poids totaux (%)")
    plt.show()


    plt.figure()
    plt.hist(lee1)
    plt.title("Erreurs essieu 1 (t)")

    plt.figure()
    plt.hist(lee2)
    plt.title("Erreurs essieu 2 (t)")


    plt.figure()
    plt.hist(lee3)
    plt.title("Erreurs essieu 3 (t)")


    plt.figure()
    plt.hist(lee4)
    plt.title("Erreurs essieu 4 (t)")


    plt.figure()
    plt.hist(lee5)
    plt.title("Erreurs essieu 5 (t)")

    plt.show()
