import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pad
from scipy.stats import linregress
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






#Verification de code partie D et E
pad.options.display.float_format = "{:.4f}".format  # Set 4 decimal places
def print_matrix(matrix):
    df = pad.DataFrame(matrix)
    print(df)

# Paramètres
D = 1
R = D/2
Ce = 20
S = 2*(10**(-8))
D = 10**(-10)



# Fonction analytique
def C(r):
    return (0.25*S*(1/D)*R**2*(((r**2/R**2)-1))) + Ce

nodes_list = np.linspace(3,403,400)  # Différentes valeurs de noeuds, avec un nombre minimal de noeuds de 3
h_values = []
errors_L1_D = []
errors_L2_D = []
errors_Linf_D = []
errors_L1_E = []
errors_L2_E = []
errors_Linf_E = []

for nodes in nodes_list:
    nodes = int(nodes)
    delta_r = R/(nodes-1)
    h_values.append(delta_r)
    
    # Initialisation des matrices
    A = np.zeros((nodes, nodes))
    A_QE = np.zeros((nodes, nodes))
    b = np.zeros(nodes)
    
    ## Construction de A
    for i in range(1, nodes-1):
        A[i, i-1] = 1/(delta_r**2)
        A[i, i] = -2/(delta_r**2)-(1/(i*delta_r**2))
        A[i, i+1] = 1/(delta_r**2)+(1/(i*delta_r**2))
    
    A[0, 0] = -3/(2*delta_r)
    A[0, 1] = 4/(2*delta_r)
    A[0, 2] = -1/(2*delta_r)
    A[nodes-1, nodes-1] = 1
    
    for i in range(1, nodes-1):
        b[i] = S/D
    b[0] = 0
    b[nodes-1] = Ce
    
    concentration = np.linalg.solve(A, b)
    
    ## Construction de A_QE
    for i in range(1, nodes-1):
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
    
    concentration_QE = np.linalg.solve(A_QE, b)
    
    r_values = np.linspace(0, R, nodes)
    C_exact = np.array([C(r) for r in r_values])
    
    error_L1_D = np.sum(np.abs(concentration - C_exact))
    error_L2_D = np.sqrt(np.sum((concentration - C_exact)**2))
    error_Linf_D = np.max(np.abs(concentration - C_exact))
    
    error_L1_E = np.sum(np.abs(concentration_QE - C_exact))
    error_L2_E = np.sqrt(np.sum((concentration_QE - C_exact)**2))
    error_Linf_E = np.max(np.abs(concentration_QE - C_exact))
    
    errors_L1_D.append(error_L1_D)
    errors_L2_D.append(error_L2_D)
    errors_Linf_D.append(error_Linf_D)
    
    errors_L1_E.append(error_L1_E)
    errors_L2_E.append(error_L2_E)
    errors_Linf_E.append(error_Linf_E)

# Tracé des erreurs
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_L1_D, 'ro-', label="L1 - Question D")
plt.loglog(h_values, errors_L2_D, 'bo-', label="L2 - Question D")
plt.loglog(h_values, errors_Linf_D, 'go-', label="Linfini - Question D")


plt.xlabel("h = R / (nodes - 1)")
plt.ylabel("Erreurs")
plt.legend()
plt.grid()
plt.title("Évolution des erreurs en fonction de h question D")
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_L1_E, 'r--', label="L1 - Question E")
plt.loglog(h_values, errors_L2_E, 'b--', label="L2 - Question E")
plt.loglog(h_values, errors_Linf_E, 'g--', label="Linfini - Question E")
plt.xlabel("h = R / (nodes - 1)")
plt.ylabel("Erreurs")
plt.legend()
plt.grid()
plt.title("Évolution des erreurs en fonction de h questionb E")
plt.show()


# Fonction pour tracer les erreurs et leur régression
def plot_convergence(h_values, errors_L1, errors_L2, errors_Linf):
    plt.figure(figsize=(10, 7))

    # Liste des erreurs et couleurs associées
    errors_list = [errors_L1, errors_L2, errors_Linf]
    colors = ['b', 'g', 'r']
    labels = [r"L_1", r"L_2", r"L_{\infty}"]

    for i, errors in enumerate(errors_list):
        # Régression en loi de puissance : log-log
        coefficients = np.polyfit(np.log(h_values[5:]), np.log(errors[5:]), 1)
        exponent = coefficients[0]
        fit_function = lambda x: np.exp(coefficients[1]) * x**exponent

        # Tracé des erreurs et de la régression
        plt.scatter(h_values, errors, marker='o', color=colors[i], label=f"{labels[i]} - Données")
        plt.plot(h_values, fit_function(h_values), linestyle='--', color=colors[i], label=f"{labels[i]} - Régression")

        # Ajout de l'équation sur le graphe avec syntaxe correcte
        equation_text = fr"${labels[i]} = {np.exp(coefficients[1]):.4f} \times h^{{{exponent:.2f}}}$"
        h_min = min(h_values)
        plt.text(h_min*1.5, errors[-1], equation_text, fontsize=12, color=colors[i])

    # Format du graphique
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Taille de maille $h$', fontsize=12, fontweight='bold')
    plt.ylabel('Erreurs', fontsize=12, fontweight='bold')
    plt.title("Convergence d'ordre 2 des erreurs en fonction de $h$", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.show()

# Appel de la fonction avec tes données
plot_convergence(h_values, errors_L1_D, errors_L2_D, errors_Linf_D)
