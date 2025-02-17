import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Configuration d'affichage des nombres flottants
pd.options.display.float_format = "{:.4f}".format

def print_matrix(matrix):
    """Affiche une matrice sous forme de DataFrame pour une meilleure lisibilité."""
    df = pd.DataFrame(matrix)
    print(df)

class DiffusionParameters:
    """Classe contenant les paramètres de diffusion."""
    def __init__(self, R, Ce, S, D_eff):
        self.R = R
        self.Ce = Ce
        self.S = S
        self.D_eff = D_eff

params = DiffusionParameters(0.5, 20, 2e-8, 1e-10)

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

nodes_list = np.linspace(3, 403, 400).astype(int)
h_values = params.R / (nodes_list - 1)
errors_L1_D, errors_L2_D, errors_Linf_D = [], [], []
errors_L1_E, errors_L2_E, errors_Linf_E = [], [], []

for nodes in nodes_list:
    r_values = np.linspace(0, params.R, nodes)
    C_exact = np.array([concentration_analytique(r, params) for r in r_values])
    
    C_ordre_1 = concentration_ordre_1(nodes, params)
    C_ordre_2 = concentration_ordre_2(nodes, params)
    
    errors_L1_D.append(np.sum(np.abs(C_ordre_1 - C_exact)))
    errors_L2_D.append(np.sqrt(np.sum((C_ordre_1 - C_exact) ** 2)))
    errors_Linf_D.append(np.max(np.abs(C_ordre_1 - C_exact)))
    
    errors_L1_E.append(np.sum(np.abs(C_ordre_2 - C_exact)))
    errors_L2_E.append(np.sqrt(np.sum((C_ordre_2 - C_exact) ** 2)))
    errors_Linf_E.append(np.max(np.abs(C_ordre_2 - C_exact)))


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
plt.title("Évolution des erreurs en fonction de h question E")
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
