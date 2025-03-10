"Ce fichier contient les fonctions pour la création des graphiques"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

def plot_convergenceh(h_values, errors_L1, errors_L2, errors_Linf):
    plt.figure(figsize=(10, 7))

    # Liste des erreurs et couleurs associées
    errors_list = [errors_L1, errors_L2, errors_Linf]
    colors = ['b', 'g', 'r']
    labels = [r"L_1", r"L_2", r"L_{\infty}"]

    for i, errors in enumerate(errors_list):
        # Régression en loi de puissance : log-log
        coefficients = np.polyfit(np.log(h_values[40:]), np.log(errors[40:]), 1)
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
    plt.title("Convergence spatiale des erreurs en fonction de $h$", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.show()
    
def plot_convergencet(dt_values, errors_L1, errors_L2, errors_Linf):
    plt.figure(figsize=(10, 7))

    # Liste des erreurs et couleurs associées
    errors_list = [errors_L1, errors_L2, errors_Linf]
    colors = ['b', 'g', 'r']
    labels = [r"L_1", r"L_2", r"L_{\infty}"]

    for i, errors in enumerate(errors_list):
        # Régression en loi de puissance : log-log
        coefficients = np.polyfit(np.log(dt_values[:7]), np.log(errors[:7]), 1)
        exponent = coefficients[0]
        fit_function = lambda x: np.exp(coefficients[1]) * x**exponent

        # Tracé des erreurs et de la régression
        plt.scatter(dt_values, errors, marker='o', color=colors[i], label=f"{labels[i]} - Données")
        plt.plot(dt_values, fit_function(dt_values), linestyle='--', color=colors[i], label=f"{labels[i]} - Régression")

        # Ajout de l'équation sur le graphe avec syntaxe correcte
        equation_text = fr"${labels[i]} = {np.exp(coefficients[1]):.4f} \times h^{{{exponent:.2f}}}$"
        dt_min = min(dt_values)
        plt.text(dt_min*1.5, errors[-1], equation_text, fontsize=12, color=colors[i])

    # Format du graphique
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Pas de temps $dt$', fontsize=12, fontweight='bold')
    plt.ylabel('Erreurs', fontsize=12, fontweight='bold')
    plt.title("Convergence temporelle des erreurs en fonction de $dt$", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.show()

def plot_evolution_error(h_values, errors_L1, errors_L2, errors_Linf,titre):
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_L1, 'r--', label="L1")
    plt.loglog(h_values, errors_L2, 'b--', label="L2")
    plt.loglog(h_values, errors_Linf, 'g--', label="Linfini")

    plt.xlabel("h = R / (nodes - 1)")
    plt.ylabel("Erreurs")
    plt.legend()
    plt.grid()
    plt.title(titre)
    plt.show()

