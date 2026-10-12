"""Simulation de la dynamique d'une population de Daphnia magna.

Le modèle est un modèle de Verhulst étendu : le frein de densité est
multiplié par des facteurs environnementaux et une mortalité explicite.
La résolution temporelle utilise Euler avec dt = 0,05 jour.
"""
import numpy as np
import matplotlib.pyplot as plt

TEMPERATURE = 22.0
T_OPT = 22.0
SIGMA_T = 6.0
RMAX = 0.28                    # taux de naissance maximal, jour^-1
M_BASE = 0.03                  # mortalité de base, jour^-1
M_FAMINE = 0.32                # mortalité supplémentaire en famine, jour^-1
SEUIL_FAMINE = 0.50            # seuil relatif de famine
DENSITE_MAX_PAR_ML = 0.18      # capacité brute, daphnies/mL
APPORT_OPTIMAL_ML = 0.05       # ration calibrée sur le témoin, mL/daphnie/jour
PH_OPT = 7.5
DT = 0.05


def f_nourriture(apport_ml):
    """Calcule un facteur continu de nourriture autour de l'optimum calibré."""
    if apport_ml <= 0:
        return 0.0
    # Une ration trop faible ou trop forte réduit la reproduction; la fonction
    # reste continue sur l'intervalle d'entrée demandé.
    return float(np.exp(-((apport_ml - APPORT_OPTIMAL_ML) / 0.060) ** 2))


def f_ph(ph):
    """Calcule l'effet relatif du pH, maximal à pH 7,5 et très faible aux extrêmes."""
    return float(max(0.02, np.exp(-((ph - PH_OPT) / 2.0) ** 2)))


def f_lumiere(heures):
    """Calcule un facteur symétrique de photopériode, égal à 1 à 12 h."""
    return float(np.exp(-((heures - 12.0) / 8.0) ** 2))


def f_temperature(temperature):
    """Calcule l'effet gaussien de la température autour de 22 °C."""
    return float(np.exp(-((temperature - T_OPT) ** 2) / (2 * SIGMA_T ** 2)))


def f_densite(n, volume_ml):
    """Calcule le frein de densité max(0, 1 - N/K)."""
    k = DENSITE_MAX_PAR_ML * volume_ml
    return max(0.0, 1.0 - n / k)


def mortalite(apport_ml):
    """Calcule la mortalité totale, avec un terme de famine progressif."""
    ration = f_nourriture(apport_ml)
    famine = max(0.0, 1.0 - ration / SEUIL_FAMINE)
    return M_BASE + M_FAMINE * famine


def taux_naissance_max(apport_ml, ph, lumiere):
    """Calcule le taux de naissance avant limitation par la densité."""
    return RMAX * f_nourriture(apport_ml) * f_ph(ph) * f_lumiere(lumiere) * f_temperature(TEMPERATURE)


def simuler(n0, volume_ml, apport_ml, ph, lumiere, duree):
    """Simule l'effectif avec Euler et renvoie les vecteurs temps et population."""
    n_pas = int(round(duree / DT))
    temps = np.linspace(0, duree, n_pas + 1)
    population = np.zeros(n_pas + 1)
    population[0] = n0
    naissance_hors_densite = taux_naissance_max(apport_ml, ph, lumiere)
    taux_mortalite = mortalite(apport_ml)
    for i in range(n_pas):
        n = population[i]
        variation = n * (naissance_hors_densite * f_densite(n, volume_ml) - taux_mortalite)
        population[i + 1] = max(0.0, n + DT * variation)
    return temps, population


def seuil_maximal(volume_ml, apport_ml, ph, lumiere):
    """Calcule le seuil d'équilibre positif du modèle, sans dépasser K."""
    b0 = taux_naissance_max(apport_ml, ph, lumiere)
    m = mortalite(apport_ml)
    k = DENSITE_MAX_PAR_ML * volume_ml
    if b0 <= m:
        return 0.0
    return k * (1.0 - m / b0)


def demander(message, mini, maxi, type_valeur=float):
    """Demande une valeur contrôlée par input et la maintient dans ses bornes."""
    while True:
        texte = input(f"{message} [{mini} à {maxi}] : ").strip().replace(',', '.')
        if not texte:
            print("  Saisie obligatoire.")
            continue
        try:
            valeur = type_valeur(texte)
        except ValueError:
            print("  Valeur invalide.")
            continue
        if mini <= valeur <= maxi:
            return valeur
        print(f"  Hors limites ({mini} à {maxi}).")


def afficher_graphique(temps, population, seuil, volume, apport, ph, lumiere, duree):
    """Trace le seul graphique du programme, avec le seuil maximal K affiché."""
    plt.figure(figsize=(9, 5))
    plt.plot(temps, population, color='tab:blue', linewidth=2, label='Population simulée')
    plt.axhline(seuil, color='tab:red', linestyle='--', linewidth=1.5,
                label=f'Seuil maximal ≈ {seuil:.1f} daphnies')
    plt.xlabel('Temps (jours)'); plt.ylabel('Nombre de daphnies')
    plt.title(f'Dynamique de Daphnia magna sur {duree} jours')
    plt.grid(alpha=0.3); plt.xlim(0, duree); plt.ylim(0, max(seuil, population.max(), 1) * 1.15)
    plt.legend(loc='lower right'); plt.tight_layout(); plt.show()


def programme_principal():
    """Exécute l'interface interactive complète et affiche la simulation."""
    print(f"Référence: nourriture optimale par daphnie = {APPORT_OPTIMAL_ML:.3f} mL/jour")
    n0 = demander('Daphnies initiales', 1, 100, int)
    volume = demander('Volume (mL)', 100, 10000)
    apport = demander(f'Nourriture (mL/daphnie/jour; optimum = {APPORT_OPTIMAL_ML:.3f})', 0, 5)
    ph = demander('pH', 0, 14)
    lumiere = demander('Heures de lumière (optimum = 12 h)', 0, 24)
    duree = demander('Durée (jours)', 1, 365, int)
    temps, population = simuler(n0, volume, apport, ph, lumiere, duree)
    seuil = seuil_maximal(volume, apport, ph, lumiere)
    afficher_graphique(temps, population, seuil, volume, apport, ph, lumiere, duree)


if __name__ == '__main__':
    programme_principal()
