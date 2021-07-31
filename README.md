# Bwifsttar

Bwifsttar is a python library based on the physic model BWIM (bridge-weighing-in-motion) whose functions are mainly used for :

    - Calibration (calculation of the line of influence of a bridge from the measured signals of the heavyweights)
    
    - Weighing in motion (from the line of influence)
    
    - Calculation of the influence line with physic informed deep learning (PINN)

A set of Notebooks and a course report are also available with examples of how to use the functions and explanations of the research process around this subject.


## Installation

```bash
git clone https://git.esiee.fr/deturchf/stage_ponts_instrumentes.git
```
Within the repository : 

```bash
pip3 install -r workspace/Bwifsttar/requirements.txt
```

## Read Notebooks

```bash
pip3 install jupyter
```

```bash
jupyter notebook
```

## Usage example

```python
from Bwifsttar import load_senlis_modified,calibration_mt_reg,estimation_mt

calibration_trucks,traffic_trucks = load_senlis_modified(6) #loading of Senlis bridge data according to sensor 6

LI = calibration_mt_reg(calibration_trucks[0:8],tv_reg={'strength': 1e2, 'cutoff': 0.95}) #Calculation of the influence line with total variation regularization and with the first eight trucks
truck = calibration_trucks[0:1]
w = estimation_mt(truck, h) #weighing in motion from LI
```

## Folders and files description

### Folders 
    
    - workspace : Contains all Notebooks, folders and files linked to research (with code)
    - papers : Contains some papers about our subject
    - data : Contains the data used during this internship 
    - CRs : Contains weekly reports 
    - workspace/Bwifsttar : Contains Bwifsttar library (contains an other `README.md` for more details)
    - workspace/Data_PINN : Contains `.npy` files to load for physic informed neural network training
    - workspace/Errors : Contains results from statistical procedure for different calibration/traffic trucks repartition and cases
    
        - Errors/erreurs_dec : stats with peaks shifting
        
        - Errors/wo_dec : stats without peaks shifting
        
        - Errors/erreurs_simu : stats from simulation
        
    - workspace/Images_Notebooks : Images used in Notebooks
    - workspace/Saved : Results saved from different Notebooks
    - workspace/data : Contains theorical results 
    - workspace/script: Useful for data loading 
    
    
### Files
    
    - workspace/N0-18 : Notebooks used during the internship for research. All are "well" named,commented with plots for a good understanding
    - workspace/NS1-4 : Special Notebooks used for tests or additional research
    - workspace/erreursX/erreur_essieuY.npy : Stats for X% repartition of calibration trucks for the Yth axle
    - workspace/erreursX/erreur_tot.npy : Stats for X% repartition of calibration trucks on the total weight (can be positive or negative)
    - workspace/erreursX/erreur_tot_redr.npy : Stats for X% repartition of calibration trucks on the total weight (absolute value)

Others files (which are not described here) can be understood thanks to Notebooks. Here were the main ones.

## License
**@Copyrights, all rights reserved**
IFSTTAR

@authors

Franck Deturche Dura (intern)

Franziska Schmidt (IFSTTAR tutor)

Jean-François Bercher (ESIEE Paris tutor)

Giovanni Chierchia (ESIEE Paris tutor)

