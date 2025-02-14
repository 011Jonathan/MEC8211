import numpy as np
import matplotlib.pyplot as plt

def resolution_ordre_1(nombre_de_noeuds):
    """
    Cette fonction trouve la concentration en chaque noeud à l'aide de la méthode des différences finies.
    Entrée : 
        - nombre_de_noeuds : nombre de points de discrétisation
    Sortie : 
        - Un vecteur contenant la concentration en chaque noeud
    """
    
    # Constantes
    D_eff = 1e-10  # Coefficient de diffusion (m²/s)
    S = 2e-8       # Terme source constant (mol/m³/s)
    R = 0.5        # Rayon du pilier (m)
    Ce = 20        # Concentration imposée en surface (mol/m³)

    # Discrétisation
    dr = R / (nombre_de_noeuds - 1)  
    r = np.linspace(0, R, nombre_de_noeuds)  

    # Matrices du système linéaire
    A = np.zeros((nombre_de_noeuds, nombre_de_noeuds))
    B = np.zeros(nombre_de_noeuds)

    # Condition de Newmann : dC/dr = 0 à r = 0
    A[0, 0] = 1
    A[0, 1] = -1 
    B[0] = 0  # 

    # Condition à la surface : C = Ce
    A[-1, -1] = 1
    B[-1] = Ce

    # Remplissage de la matrice pour les points internes
    for i in range(1, nombre_de_noeuds - 1):
        A[i, i-1] = 1
        A[i, i] = -2 - dr / r[i]
        A[i, i+1] =  1 + dr / r[i]
        B[i] = S * dr**2 / D_eff

    # Résolution du système linéaire
    concentration = np.linalg.solve(A, B)
    concentration_exact = (S / (4 * D_eff)) * R**2 * ((r**2 / R**2) - 1) + Ce

    return r, concentration, concentration_exact


# test et graphique
noeuds = 5
r, C,C_exact = resolution_ordre_1(noeuds)


# Affichage des résultats
plt.plot(r, C, 'bo-', label="Solution numérique")
plt.plot(r, C_exact, 'ro--', label="Solution analytique")
plt.xlabel("Rayon r [m]")
plt.ylabel("Concentration C [mol/m³]")
plt.title("Comparaison des solutions numérique et analytique\n de la distribution de la concentration en fonction du rayon")
plt.legend()
plt.grid()
plt.show()
