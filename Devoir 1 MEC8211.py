from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pad

pad.options.display.float_format = "{:.4f}".format  # Set 4 decimal places
def print_matrix(matrix):
    df = pad.DataFrame(matrix)
    print(df)

# Paramètres
D = 1
R = D/2
nodes = 5
delta_r = R/(nodes-1)
Ce = 20
S = 2*(10**(-8))
D = 10**(-10)

# Fonction analytique
def C(r):
    return (0.25*S*(1/D)*R**2*(((r**2/R**2)-1))) + Ce

# Initialisation des matrices
A = [[0]*nodes for i in range(nodes)]   # Matrice des coefficients pour Question D
A_QE = [[0]*nodes for i in range(nodes)]   # Matrice des coefficients pour Question E
b = [0]*nodes # Matrice à droite de l'égalité, identique pour les questions D et E

### QUESTION D

## Construction de A au centre du maillage c'est à dire sans les noeuds frontières
for i in range(1,nodes-1):
    #print(i)
    A[i][i-1] = 1/(delta_r**2)
    A[i][i] = -2/(delta_r**2)-(1/(i*delta_r**2))  # En consiérant la valeur de r au noeud. Donc un multiple de delta_r
    A[i][i+1] = 1/(delta_r**2)+(1/(i*delta_r**2)) # Note : En python les énumérations commencent à 0 et pas 1

# Ajout des facteurs des noeuds aux frontières
# 1er noeud 
A[0][0] = -3/(2*delta_r)
A[0][1] = 4/(2*delta_r)
A[0][2] = -1/(2*delta_r)

# Dernier noeud
A[nodes-1][nodes-1] = 1

## Construction de b au centre du maillage c'est à dire sans les noeuds frontières
for i in range(1,nodes):
    b[i] = S/D

# Ajout des facteurs des noeuds aux frontières
b[0] = 0
b[nodes-1] = Ce

# Résultats
concentration = np.linalg.solve(A,b)
print("Les concentrations de la questions D sont :")
print_matrix(concentration)


### QUESTION E

## Construction de A_QE au centre du maillage c'est à dire sans les noeuds frontières
for i in range(1,nodes-1):
    #print(i)
    A_QE[i][i-1] = 1/(delta_r**2)-(1/(2*i*delta_r**2))
    A_QE[i][i] = -2/(delta_r**2)  
    A_QE[i][i+1] = 1/(delta_r**2)+(1/(2*i*delta_r**2)) 

# Ajout des facteurs des noeuds aux frontières
# 1er noeud 
A_QE[0][0] = -3/(2*delta_r)
A_QE[0][1] = 4/(2*delta_r)
A_QE[0][2] = -1/(2*delta_r)

# Dernier noeud
A_QE[nodes-1][nodes-1] = 1

# Résultats_QE
concentration_QE = np.linalg.solve(A_QE,b)
print("Les concentrations de la questions E sont :")
print_matrix(concentration_QE)
print_matrix(A_QE)

# Analytique
xplot = np.linspace(0,R,50)
yplot = [C(x) for x in xplot]
plt.plot(xplot,yplot, label = "Analytique") 
#  Numérique
xplot2 = np.linspace(0,R,nodes)
plt.plot(xplot2,concentration,'ro',label = " Question D")
plt.plot(xplot2,concentration_QE,'go', label="Question E")
plt.xlabel("R discrétisé en 5 points")
plt.ylabel("Concentration")
plt.legend()
plt.show()