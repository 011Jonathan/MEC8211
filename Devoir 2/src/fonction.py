"Ce fichier contient les fonctions de calcul selon différentes méthodes"

import numpy as np
import paramètres as params

def concentration_analytique(r, params):
    """Calcule la concentration analytique."""
    return (0.25 * params.S * (1 / params.D_eff) * params.R ** 2 * (((r ** 2 / params.R ** 2) - 1))) + params.Ce

def concentration_ordre_1(nodes, params):
    """Calcule la concentration avec une discrétisation d'ordre 1."""
    R, Ce, S, D_eff = params.R, params.Ce, params.S, params.D_eff
    delta_r = R / (nodes - 1)
    
    A = np.zeros((nodes, nodes))
    b = np.zeros(nodes)
    
    for i in range(1, nodes - 1):
        A[i, i - 1] = 1 / (delta_r ** 2)
        A[i, i] = -2 / (delta_r ** 2) - (1 / (i * delta_r ** 2))
        A[i, i + 1] = 1 / (delta_r ** 2) + (1 / (i * delta_r ** 2))
    
    A[0, 0] = -3 / (2 * delta_r)
    A[0, 1] = 4 / (2 * delta_r)
    A[0, 2] = -1 / (2 * delta_r)
    A[nodes - 1, nodes - 1] = 1
    
    for i in range(1, nodes - 1):
        b[i] = S / D_eff
    b[nodes - 1] = Ce
    
    return np.linalg.solve(A, b)

def concentration_ordre_2(nodes, params):
    """Calcule la concentration avec une discrétisation d'ordre 2."""
    R, Ce, S, D_eff = params.R, params.Ce, params.S, params.D_eff
    delta_r = R / (nodes - 1)
    
    A = np.zeros((nodes, nodes))
    b = np.zeros(nodes)
    
    for i in range(1, nodes - 1):
        A[i, i - 1] = 1 / (delta_r ** 2) - (1 / (2 * i * delta_r ** 2))
        A[i, i] = -2 / (delta_r ** 2)
        A[i, i + 1] = 1 / (delta_r ** 2) + (1 / (2 * i * delta_r ** 2))
    
    A[0, 0] = -3 / (2 * delta_r)
    A[0, 1] = 4 / (2 * delta_r)
    A[0, 2] = -1 / (2 * delta_r)
    A[nodes - 1, nodes - 1] = 1
    
    for i in range(1, nodes - 1):
        b[i] = S / D_eff
    b[nodes - 1] = Ce
    
    return np.linalg.solve(A, b)