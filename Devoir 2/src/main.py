"Fichier pour l'exécution du projet"

import fonction as fonction
import paramètres as params
import visualisation as graph
from math import *
import numpy as np
import matplotlib.pyplot as plt

"""Question b) MMS"""
r = np.linspace(0,params.R,100)
t = np.linspace(0,31536000 * 100,100)

r_mms, t_mms = np.meshgrid(r,t)
C_mms = fonction.MMS(r_mms,t_mms)

plt.contourf(r_mms,t_mms,C_mms)
plt.colorbar()
plt.title('Fonction MMS')
plt.xlabel('Position r')
plt.ylabel('temps')
plt.show()

"""Question c) terme source"""

r_source, t_source = np.meshgrid(r,t)
source = fonction.Terme_source(r_source,t_source)

plt.contourf(r_source,t_source,source)
plt.colorbar()
plt.title('Terme source')
plt.xlabel('Position r')
plt.ylabel('temps')
plt.show()

"""
nodes_list = np.linspace(3, 403, 400).astype(int)
h_values = params.R / (nodes_list - 1)
errors_L1_D, errors_L2_D, errors_Linf_D = [], [], []
errors_L1_E, errors_L2_E, errors_Linf_E = [], [], []

for nodes in nodes_list:
    r_values = np.linspace(0, params.R, nodes)
    C_exact = np.array([fonction.concentration_analytique(r, params) for r in r_values])
    
    C_ordre_1 = fonction.concentration_ordre_1(nodes, params)
    C_ordre_2 = fonction.concentration_ordre_2(nodes, params)
    
    errors_L1_D.append(np.sum(np.abs(C_ordre_1 - C_exact)))
    errors_L2_D.append(np.sqrt(np.sum((C_ordre_1 - C_exact) ** 2)))
    errors_Linf_D.append(np.max(np.abs(C_ordre_1 - C_exact)))
    
    errors_L1_E.append(np.sum(np.abs(C_ordre_2 - C_exact)))
    errors_L2_E.append(np.sqrt(np.sum((C_ordre_2 - C_exact) ** 2)))
    errors_Linf_E.append(np.max(np.abs(C_ordre_2 - C_exact)))

graph.plot_evolution_error(h_values, errors_L1_D, errors_L2_D, errors_Linf_D,"Évolution des erreurs en fonction de h question D")
graph.plot_evolution_error(h_values, errors_L1_E, errors_L2_E, errors_Linf_E,"Évolution des erreurs en fonction de h question E")
graph.plot_convergence(h_values, errors_L1_D, errors_L2_D, errors_Linf_D)
"""


"Question f) du devoir 2"

N_tot = 11
t = 4e9 
t_mois = int(t / (60*60*24*365/12))

result,r_source,time_line = fonction.concentration(N_tot,t_mois,6)

plt.figure(figsize=(10, 7))
for nodes in range(N_tot):
    plt.plot(time_line, [ligne[nodes] for ligne in result],label="noeuds "f"{nodes}")

plt.xlabel("Temps [mois]", fontsize=12, fontweight='bold')
plt.ylabel('Concentration [mol/m^3]', fontsize=12, fontweight='bold')
plt.title("Évalution de la concentration en fonction du temps pour 11 noeuds", fontsize=14, fontweight='bold')
plt.legend(loc = 'upper right')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.show()