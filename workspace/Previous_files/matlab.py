import scipy.io

#!ls ../data
mat = scipy.io.loadmat('../data/SimulatedData.mat')

#informations sur les camions et le pont
"""
for i in range(0,50):
    val = mat['Data'][0][0][3][i]
    print(val)
    print("Type du pont :" ,val[0][0][0][0])
    print("Nombre Essieux :" ,val[0][0][0][1])
    print("Position Essieux :" ,val[0][0][0][2])
    print("Poids Essieux :" ,val[0][0][0][3])
    print("Vélocité :" ,val[0][0][0][4])
    print("Kv :" ,val[0][0][0][5])
    print("Cv :" ,val[0][0][0][6])


    print("\n")
"""
"""
bridge = mat['Data'][0][0]['bridge']
print("Bridge : ",bridge)
print(bridge.dtype)


settings = mat['Data'][0][0]['Settings']
print("Settings :",settings)
print(settings.dtype)

nVehicles = mat['Data'][0][0]['nVehicles']
print("nVehicles",nVehicles)
print(nVehicles.dtype)

Vehicles = mat['Data'][0][0]['Vehicles']
print("Vehicles : ",Vehicles)
print(Vehicles.dtype)

FrequencyVehicles = mat['Data'][0][0]['FrequencyVehicles']
print("FrequencyVehicles : ",FrequencyVehicles)
print(FrequencyVehicles.dtype)
"""
Cases = mat['Data'][0][0]['Cases']
print("Cases : ",Cases)
print(Cases.dtype)
#val
