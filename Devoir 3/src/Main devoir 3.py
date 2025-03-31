# CE SCRIPT PERMET DE TRACER LA PDF ET LA CDF DES LOG DE LA PERMÉABILITÉ
# IL PERMET AUSSI DE CALCULER U INPUT, UNUM ET UD POUR ENCADRER DELTA_MODEL


#Import de librairies utiles
from math import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pad

#Formatage de l'afficahge matricieelle
pad.options.display.float_format = "{:.4f}".format  # Set 4 decimal places
def print_matrix(matrix):
    df = pad.DataFrame(matrix)
    print(df)

# Lecture d'un fichier excell contanant les valeurs de perméabilité obtenue dans Matlab
# Vous pouvez remplacer le chemein par la localisation du fichier excell sur votre ordinateur
fichier_excel = "C:/.../Devoir3-extract.xlsx"  
df = pad.read_excel(fichier_excel)

# Ici k2 est le l'entete de la colone ou se trouve les valeurs de perméabilité
colonne = "k2"  
liste = df[colonne].tolist()  # Formate les valeurs de la colone en liste

# Calcul des logs de perméabilité
val_k=[log(x) for x in liste]

#Calcul de la moyenne des log de perrméabilité
moyenne_per = sum(val_k)/len(val_k)
print("moyenne_per =", moyenne_per)

# Calcul de la variance des log de perméabilité
somme_caré_écart = sum((x-moyenne_per)**2 for x in val_k)
variance_per = somme_caré_écart/(len(val_k)-1)
print("Variance_per =", variance_per)

# Calcul de l'écart type
ecart_type_per = variance_per**0.5
print("ecart_type_per =", ecart_type_per)
#print("exp2sigma=",exp(moyenne_per+2*ecart_type_per)-exp(moyenne_per-2*ecart_type_per))

# Calcul de u_input en rapportant le FVG
u_input = exp(ecart_type_per)
print("u_input =",u_input)

# Traçage de l'histogramme des valeurs et de la PDF
plt.hist(val_k,25,density=True)
plt.title("Histogramme et PDF de log(perméabilité)")
plt.xlabel("log(Perméabilité)")
plt.ylabel("Nombre")

xmin, xmax = moyenne_per-4*ecart_type_per, moyenne_per+4*ecart_type_per
lnspc = np.linspace(xmin, xmax, len(val_k))
fit_moyenne, fit_ecart_type = stats.norm.fit(val_k)

# superposer la pdf
pdf = stats.norm.pdf(lnspc, fit_moyenne, fit_ecart_type)
label = "Moyenne = {:.2f}".format(fit_moyenne) + "\nEcart-type = {:.2f}".format(fit_ecart_type)
plt.plot(lnspc, pdf, label=label)
plt.legend()
plt.show()

# Traçage de l'histogramme cumulatif et la cumulative distribution function (cdf)
plt.hist(val_k, 20, cumulative=True, density=True)
plt.title("Histogramme cumulatif et CDF de log(perméabilité)")
plt.xlabel("log(Perméabilité)")
plt.ylabel("Probabilité < log(perméabilité)")
cdf = stats.norm.cdf(lnspc, fit_moyenne, fit_ecart_type)
plt.plot(lnspc, cdf, label="Norm")
plt.legend()
plt.show()

# Écart type des incertitudes expérimentales
sigma_fibre = 2.85
sigma_poro = 7.5*10**(-3)
sigma_perme = 14.7
sigma_manu = 10

# Calcul de u_D
u_D = (sigma_perme**2+sigma_poro**2+sigma_fibre**2+sigma_manu**2)**0.
print("u_D =",u_D)

#Calcul de E
E = exp(moyenne_per)-80.6
print("E =", E)

# Vaeleur de unum trouver avec le GCI
u_num = 0.075

# Calcul de u_val
u_val = (u_input**2+u_num**2+u_D**2)**0.5
print("u*_val",u_val)

# Calcul des bornes de delta_model avec k =2 pour un intervalle de confiance à 95.4%
borne_inf = E-2*u_val
borne_sup = E+2*u_val
print("Les bornes sont:", borne_inf, borne_sup)

# Calcul du pourcentage de 2*u_val sur la valeur absolue de E
print("%_ge de Uval sur E",2*u_val*100/abs(E))