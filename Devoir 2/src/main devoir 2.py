# Ce code résout l'équation de variation de la concentration dans un pilier de béton, 
# en utilisant une méthode matricielle et la solution exacte (MMS) pour comparaison.
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pad


pad.options.display.float_format = "{:.4f}".format  

# Fonction pour afficher une matrice sous forme de DataFrame
def print_matrix(matrix):
    df = pad.DataFrame(matrix)
    print(df)



# Paramètres principaux
nodes = 5  # Nombre de nœuds pour le maillage
time_step = 31536000  # Durée d'un pas de temps (1 an en secondes)
k = 4*(10**(-9))  # Coefficient de réaction
Deff = 10**(-10)  # Diffusivité effective
D = 1  # Diamètre
R = D/2  # Rayon
time_factor = 126  # Facteur de temps pour la simulation

# Fonction de résolution matricielle
def resolution_matricielle(nodes, time_step, k, Deff, R, time_factor):
    # Paramètres de calcul
    delta_r = R/(nodes-1)  # Pas spatial
    ce = 20  # Condition frontière au dernier noeuds
    
    time_wanted = time_step * time_factor  # Temps total de la simulation
    time_done = 0  # Temps écoulé
    c_initiale = 0  # Concentration initiale
    count = 0  # Compteur d'itérations

    # Initialisation des résultats (pour la concentration)
    result = [[c_initiale]*nodes]
    result[0][nodes-1] = ce

    # Initialisation des résultats en calculant avec MMS
    result_MMS = [[0]*nodes]
    for i in range(0, nodes):
        result_MMS[0][i] = (R**2 - (i * delta_r)**2) + 1
    result_MMS[0][nodes-1] = 1

    # Initialisation de la matrice des coefficients A
    A = [[0]*nodes for i in range(nodes)] 

    time_line = []  # Liste pour stocker les instants de temps
    time_line.append(0)

    # Boucle pour résoudre l'équation en itérant sur le temps
    while time_done < time_wanted:
        time_done += time_step  # Mise à jour du temps écoulé
        time_line.append(time_done)  # Ajout du temps à la liste
        count += 1  # Incrémentation du compteur d'itérations
        
        # Construction de la matrice A  sans les nœuds frontières
        for i in range(1, nodes-1):
            A[i][i-1] = (time_step*Deff) / (2*delta_r*i*delta_r) - (time_step*Deff) / (delta_r**2)
            A[i][i] = (2*time_step*Deff) / (delta_r**2) + 1 + time_step*k  # Prend en compte la valeur de r au nœud
            A[i][i+1] = (-time_step*Deff) / (2*delta_r*i*delta_r) - (time_step*Deff) / (delta_r**2)

        # Traitement des conditions aux frontières (premier et dernier nœud)
        A[0][0] = -3  # Première condition aux frontières
        A[0][1] = 4
        A[0][2] = -1
        A[nodes-1][nodes-1] = 1  # Dernière condition aux frontières

        # b au partie de droite de l'égalité
        b = [x for x in result[count-1]]  # Concentration au pas de temps précédent
        b[nodes-1] = ce  # Condition frontière pour le dernier nœud
        b[0] = 0  # Condition frontière pour le premier nœud

        # Construction du terme source 
        # def Terme_source(x, y):
        #     return -(10**(-8))*(R**2 - x**2) / ((1 + 10**(-8) * y)**2) + (4*Deff) / (1 + 10**(-8) * y) + k * ((R**2 - x**2) * (1 / (1 + 10**(-8) * y)) + 1)
        
        # b_source sert à mettre à jour la partie de droite de l'égalité avec le modèle MMS
        b_source = []
        j = -1
        for i in result_MMS[count-1]:
            j += 1
            val = Terme_source((j * delta_r), time_done)  # Calcul du terme source
            b_source.append(i + val * time_step)  # Ajout au vecteur b_source
        b_source[nodes-1] = 1  # Condition frontière
        b_source[0] = 0  # Condition frontière

        # Résolution du système linéaire pour obtenir la concentration
        concentration = np.linalg.solve(A, b)  # Résolution matricielle
        result = np.vstack((result, concentration))  # Ajout du résultat à la liste
        concentration_MMS = np.linalg.solve(A, b_source)  # Résolution pour MMS
        result_MMS = np.vstack((result_MMS, concentration_MMS))  # Ajout du résultat MMS

    return result_MMS

# Fonction MMS (Solution exacte de la concentration)
def MMS(x, y):
    return (R**2 - x**2) * (1 / (1 + 10**(-8) * y)) + 1

# Initialisation des vecteurs pour les positions et les temps
x_0 = np.linspace(0, R, nodes)
y_0 = np.linspace(0, time_step * 126, 127)
x_mms, y_mms = np.meshgrid(x_0, y_0)
z_mms = MMS(x_mms, y_mms)

# Terme source
def Terme_source(x, y):
    return -(10**(-8))*(R**2 - x**2) / ((1 + 10**(-8) * y)**2) + (4 * Deff) / (1 + 10**(-8) * y) + k * ((R**2 - x**2) * (1 / (1 + 10**(-8) * y)) + 1)

# Calcul de l'erreur entre la solution numérique et la solution exacte (MMS)
def calcul_erreur(nodes, time_step, k, Deff, R, time_factor):
    val_L1t, val_L2t, val_Linft = 0, 0, 0
    val_L1s, val_L2s, val_Linfs = 0, 0, 0
    L1t = []
    L2t = []
    Linft = []
    L1s = []
    L2s = []
    Linfs = []
    
    # Calcul des erreurs pour les différentes configurations de nœuds et pas de temps
    for j in nodes: # fixe le pas de temps à la valeur minimale et itère sur les noeuds
        result_MMS = resolution_matricielle(j, time_step[0], k, Deff, R, time_factor)
        x_0 = np.linspace(0, R, j)
        y_0 = np.linspace(0, time_step[0] * time_factor, time_factor + 1)
        x_mms, y_mms = np.meshgrid(x_0, y_0)
        z_mms = MMS(x_mms, y_mms)
        
        # Calcul des erreurs L1, L2, et Linf pour la concentration
        for i in range(0, len(result_MMS)):
            val_L1t += np.sum(np.abs(result_MMS[i] - z_mms[i])) / j
        L1t.append(val_L1t / time_factor)

        for i in range(0, len(result_MMS)):
            val_L2t += np.sqrt(np.sum((result_MMS[i] - z_mms[i]) ** 2) / j)
        L2t.append(val_L2t / time_factor)

        for i in range(0, len(result_MMS)):
            val_Linft += np.max(np.abs(result_MMS[i] - z_mms[i]))
        Linft.append(val_Linft / time_factor)

    for i in time_step: # Fixe le nombre de noeuds au maximum de sa liste et itère sur le pas de temps
        result_MMS = resolution_matricielle(nodes[-1], i, k, Deff, R, time_factor)
        x_0 = np.linspace(0, R, nodes[-1])
        y_0 = np.linspace(0, i * time_factor, time_factor + 1)
        x_mms, y_mms = np.meshgrid(x_0, y_0)
        z_mms = MMS(x_mms, y_mms)

        for i in range(0, len(result_MMS)):
            val_L1s += np.sum(np.abs(result_MMS[i] - z_mms[i])) / nodes[-1]
        L1s.append(val_L1s / time_factor)

        for i in range(0, len(result_MMS)):
            val_L2s += np.sqrt(np.sum((result_MMS[i] - z_mms[i]) ** 2) / nodes[-1])
        L2s.append(val_L2s / time_factor)

        for i in range(0, len(result_MMS)):
            val_Linfs += np.max(np.abs(result_MMS[i] - z_mms[i]))
        Linfs.append(val_Linfs / time_factor)

    # Compilation des résultats
    L1t = [np.sum(x) for x in L1t]
    L2t = [np.sum(x) for x in L2t]
    L1s = [np.sum(x) for x in L1s]
    L2s = [np.sum(x) for x in L2s]
    Linft = [np.sum(x) for x in Linft]
    Linfs = [np.sum(x) for x in Linfs]

    sortie = np.vstack((L1t, L2t, Linft, L1s, L2s, Linfs))
    return sortie

# Calcul des erreurs pour une gamme de valeurs de nœuds et pas de temps
noeuds_erreur = np.linspace(100, 150, 5)
noeuds_erreur_int = [int(x) for x in noeuds_erreur]
pas_erreur = np.linspace(1000, 10000, 5)
errors = calcul_erreur(noeuds_erreur_int, pas_erreur, k, Deff, R, time_factor)

# Affichage des résultats d'erreur
plt.figure(figsize=(10, 6))
plt.loglog(noeuds_erreur_int, errors[0], 'r*', label="L1 en variant delta r")
plt.loglog(noeuds_erreur_int, errors[1], 'bo', label="L2 en variant delta r")
plt.loglog(noeuds_erreur_int, errors[2], 'go', label="Linfini en variant delta r")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog(noeuds_erreur_int, errors[3], 'r*', label="L1 en variant delta t")
plt.loglog(noeuds_erreur_int, errors[4], 'bo', label="L2 en variant delta t")
plt.loglog(noeuds_erreur_int, errors[5], 'go', label="Linfini en variant delta t")
plt.legend()
plt.show()


# MMS 
plt.contourf(x_mms,y_mms,z_mms)
plt.colorbar()
plt.title('Fonction MMS')
plt.xlabel('Position r')
plt.ylabel('temps')
plt.show()

# résultats numériques
result_MMS = resolution_matricielle(nodes,time_step,k,Deff,R,time_factor)
plt.contourf(x_mms,y_mms,result_MMS)
plt.colorbar()
plt.title('MMS numérique')
plt.xlabel('Position r')
plt.ylabel('temps')
plt.show()

# Termes sources
x_source, y_source = np.meshgrid(x_0, y_0)
z_source = Terme_source(x_source,y_source)
plt.contourf(x_source,y_source,z_source)
plt.colorbar()
plt.title('Terme source')
plt.xlabel('Position r')
plt.ylabel('temps')
plt.show()