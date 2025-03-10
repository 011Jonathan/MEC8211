"Ce fichier contient les fonctions de calcul selon différentes méthodes"
from math import *
import numpy as np
import paramètres as params
import visualisation as graph
import fonction as fonction

def concentration_analytique(r, params):
    """Calcule la concentration analytique en régime permanent"""
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
        result_source :  matrice contenant la concentration à tous les noeuds pour tous les pas de temps lors de la rsolution avec le terme source de la MMS
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
    B_source = np.zeros(nodes)
    C = np.zeros(nodes)
    result = [[0]*nodes]
    result_source = [[0]*nodes]
    time_line = [0]

    A[0, 0] = -3 
    A[0, 1] = 4 
    A[0, 2] = -1 
    
    A[nodes - 1, nodes - 1] = 1

    for i in range(1, nodes - 1):
        A[i, i - 1] = (-D_eff * dt) / (delta_r ** 2) + ((D_eff * dt) / (2 * delta_r * i * delta_r))
        A[i, i] = (2 * D_eff  *dt) / (delta_r ** 2) + k * dt + 1
        A[i, i + 1] = (-D_eff * dt) / (delta_r ** 2) - ((D_eff * dt) / (2 * delta_r * i * delta_r))
    
    t = 0
    for j in range(0,t_wanted,dt):
        
        for i in range(1, nodes - 1):
            B[i] = C[i]
        B[nodes - 1] = Ce

        for i in range(1,nodes-1):
            B_source[i] = (1 + time_line[t] * k) * (R**2 - delta_r * i) - D_eff * (320 - 4 * time_line[t]) + 80 * k * (delta_r * i)**2
        B_source[nodes - 1] = Ce


        C = np.linalg.solve(A, B)
        result = np.vstack((result,C))

        C_source = np.linalg.solve(A,B_source)
        result_source = np.vstack((result_source,C_source))

        time_line.append(int(j)) #en mois
        t += 1
    
    return result, result_source, time_line

def MMS(r,t):
    """Solution manufacturée"""
    return ((80 * r**2) - t * (r**2 - params.R**2))

def Terme_source(r,t):
    """Terme source"""
    return (params.k * 80 * r**2) + (params.R**2 - r**2) * ( 1 + t * params.k) - params.D_eff * (320 - 4 * t)

def calculer_erreurs_L1_L2(C_numerique, C_exact):
    """Calcule les normes L1 et L2 par double sommation en temps et espace"""
    error_L1 = 0
    error_L2 = 0

    # Itérer sur chaque point d'espace et chaque pas de temps
    for t in range(C_numerique.shape[0]):  # Itérer sur les temps
        for r in range(C_numerique.shape[1]):  # Itérer sur les points d'espace
            error_L1 += np.abs(C_numerique[t, r] - C_exact[t, r])
            error_L2 += (C_numerique[t, r] - C_exact[t, r]) ** 2

    error_L2 = np.sqrt(error_L2)  # Calculer L2

    return error_L1, error_L2
def calculer_erreur_Linf(C_numerique, C_exact):
    """Calcule la norme infini en identifiant le maximum en temps et espace"""
    error_Linf = np.max(np.abs(C_numerique - C_exact))
    return error_Linf


def Convergence_espace():
    """Creation des matrices d'erreurs pour un pas temporel constant et un raffinement du pas spatial et appel de la fonction de visualisation"""
    nodes_list = np.linspace(3, 103, 50).astype(int)
    h_values = params.R / (nodes_list - 1)
    errors_L1_D, errors_L2_D, errors_Linf_D = [], [], []

    for nodes in nodes_list:
        r_values = np.linspace(0, params.R, nodes)  # Points d'espace
        t_values = 10  # Temps fixé en mois

        # Résultats numériques
        C_numerique, _, _ = fonction.concentration(nodes, t_values, 0.1)  # Résultats numériques

        # Calcul de la solution exacte pour tous les points d'espace et tous les temps
        # On ajuste C_exact pour qu'il ait la même forme que C_numerique
        C_exact = np.zeros_like(C_numerique)  # Créer une matrice de même forme que C_numerique

        for t in range(t_values):  # Boucle sur le temps
            for r_idx, r in enumerate(r_values):  # Boucle sur les points d'espace
                C_exact[t, r_idx] = fonction.MMS(r, t)  # Solution exacte à chaque point d'espace et chaque temps

        # Calcul des erreurs
        error_L1, error_L2 = calculer_erreurs_L1_L2(C_numerique, C_exact)
        error_Linf = calculer_erreur_Linf(C_numerique, C_exact)

        # Stockage des erreurs
        errors_L1_D.append(error_L1)
        errors_L2_D.append(error_L2)
        errors_Linf_D.append(error_Linf)
        
    errors_L1_D.reverse()
    errors_L2_D.reverse()
    errors_Linf_D.reverse()

    # Visualisation de la convergence
    graph.plot_convergenceh(h_values, errors_L1_D, errors_L2_D, errors_Linf_D)




    
def Convergence_temps():
    """Creation des matrices d'erreurs pour un pas spatial constant et un raffinement du pas temporel et appel de la fonction de visualisation"""
    dt_list = np.logspace(-3, -1, 15)  # Liste des pas de temps à tester
    t_values = 5  # Temps total (en mois)
    errors_L1_T, errors_L2_T, errors_Linf_T = [], [], []  # Listes pour stocker les erreurs

    for dt in dt_list:
        # Résultats numériques
        C_numerique, _, _ = fonction.concentration(100, t_values, dt)  # Résultats numériques

        # Création de C_exact avec la même forme que C_numerique
        C_exact = np.zeros_like(C_numerique)  # Créer une matrice de même forme que C_numerique

        r_values = np.linspace(0, params.R, 100)  # Points d'espace

        for t in range(t_values):  # Boucle sur les temps
            for r_idx, r in enumerate(r_values):  # Boucle sur les points d'espace
                C_exact[t, r_idx] = fonction.MMS(r, t)  # Solution exacte à chaque point d'espace et chaque temps

        # Calcul des erreurs
        error_L1, error_L2 = calculer_erreurs_L1_L2(C_numerique, C_exact)
        error_Linf = calculer_erreur_Linf(C_numerique, C_exact)

        # Stockage des erreurs
        errors_L1_T.append(error_L1)
        errors_L2_T.append(error_L2)
        errors_Linf_T.append(error_Linf)
        
    errors_L1_T.reverse()
    errors_L2_T.reverse()
    errors_Linf_T.reverse()
    # Visualisation de la convergence
    graph.plot_convergencet(dt_list, errors_L1_T, errors_L2_T, errors_Linf_T)

