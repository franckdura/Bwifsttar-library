def estimation(truck, influence):
    from Previous_files.bwim import create_toeplitz
    import numpy as np
    toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
    H_matrix = toeplitz @ influence
    w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals.T, rcond=None)
    return w

def estimation_mt(trucks, influence_bundle):
    """
        Données:
            - trucks : Liste de namedTuple Truck
            - influence_bundle : LI
    """
    import numpy as np
    from Bwifsttar import time_to_meter_sampling_mt
    from Previous_files.bwim import create_toeplitz
    
    list_w = []
    trucks = [truck for truck in trucks if(len(truck.weights) == len(truck.peaks))]
    for truck in trucks:
        
        influence= time_to_meter_sampling_mt(truck, influence_bundle)
        toeplitz = create_toeplitz(truck.signals.shape[-1], influence.size, truck.peaks)
        H_matrix = toeplitz @ influence
        w ,_,_,_ = np.linalg.lstsq(H_matrix.T, truck.signals, rcond=None)
        list_w.append(w)
    return list_w

def compare_weights_mt(trucks,list_estimated):
    """
        Données :
            - list_estimated : Liste des poids estimés
            - trucks : Liste des camions associés
        Sorties :
            - errors : Liste des listes des erreurs par essieu 
        Fonction : Affiche les erreurs par essieu et les stocks dans la variable errors
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    errors = np.array([])

    for i,estimated in enumerate(list_estimated):
        
        index = np.arange(len(estimated)) + 1
        bar_width = 0.45
        ver_shift = 1
        opacity = 0.8
        error = np.abs(estimated - trucks[i].weights).sum()
        print("Vitesse du camion ci dessous en km/h : ",trucks[i].speed*3.6)
        plt.figure(figsize=(9,5))
        plt.bar(index-bar_width/2, estimated,    bar_width, alpha=opacity, color='b', label='Pesées en marche')
        plt.bar(index+bar_width/2, trucks[i].weights, bar_width, alpha=opacity, color='r', label='Pesées statiques')
        plt.title('Erreur total: {:2.2f} t'.format(error), fontsize=14)
        plt.xlabel('Essieu', fontsize=14)
        plt.ylabel('Poid', fontsize=14)
        plt.legend(fontsize=14)
        for x, y, z in zip(index, trucks[i].weights, estimated):
            plt.text(x+bar_width/2, y-ver_shift, '%.2f' % y, fontsize=12, fontweight='bold', color='white', ha='center')
            plt.text(x-bar_width/2, z-ver_shift, '%.2f' % z, fontsize=12, fontweight='bold', color='white', ha='center')
        plt.show()
        errors = np.append(errors,error)
    return errors