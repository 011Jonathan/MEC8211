"Fichier pour l'exécution du projet"

import fonction as fonction
import paramètres as params
import visualisation as graph
import numpy as np

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
