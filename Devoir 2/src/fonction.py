"Ce fichier contient les fonctions de calcul selon différentes méthodes"

import numpy as np
import paramètres as params

def concentration_analytique(r, params):
    """Calcule la concentration analytique."""
    return (0.25 * params.S * (1 / params.D_eff) * params.R ** 2 * (((r ** 2 / params.R ** 2) - 1))) + params.Ce

def concentration(nodes,t_wanted,dt):
    """
    Fonction pour calculer la concentration en chaque point pour tous les pas de temps
    Entrées:
        nodes : nombre de noeuds dans le maillage [int]
        t_wanted : temps de l'analyse [mois]
        dt : pas de temps pour les itération [mois]

    Sortie:
        result : matrice contenant la concentration à tous les noeuds pour tous les pas de temps 
        time_line : vecteur contenant les différents temps    
    """

    R = params.R
    Ce = params.Ce
    D_eff = params.D_eff
    k = params.k


    t_wanted  = int(t_wanted * (60*60*24*365/12)) #Conversion de t_wanted en secondes
    dt = int(dt * (60*60*24*365/12)) #Conversion de dt en secondes

    delta_r = R / (nodes - 1)
    
    A = np.zeros((nodes, nodes))
    B = np.zeros(nodes)
    C = np.zeros(nodes)
    result = [[0]*nodes]
    time_line = [0]

    A[0, 0] = -3 
    A[0, 1] = 4 
    A[0, 2] = -1 
    
    A[nodes - 1, nodes - 1] = 1

    for i in range(1, nodes - 1):
        A[i, i - 1] = (-D_eff * dt) / (delta_r ** 2) + ((D_eff * dt) / (2 * delta_r * i * delta_r))
        A[i, i] = (2 * D_eff  *dt) / (delta_r ** 2) + k * dt + 1
        A[i, i + 1] = (-D_eff * dt) / (delta_r ** 2) - ((D_eff * dt) / (2 * delta_r * i * delta_r))
    
    for j in range(0,t_wanted,dt):
        
        for i in range(1, nodes - 1):
            B[i] = C[i]
        B[nodes - 1] = Ce
        
        C = np.linalg.solve(A, B)
        result = np.vstack((result,C))
        time_line.append(int(j/2628000)) #en mois
    
    return result, time_line