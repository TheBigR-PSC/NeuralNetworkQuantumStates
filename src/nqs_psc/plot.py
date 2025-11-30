import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import textwrap


# on trace tous les run d'un dossier qui contient plusieurs couple logger,metadonnées
#!!! il est impératif que le dossier en question ne contienne que des sous-dossiers de type 'model=...'


def get_sub_files(
    nom_du_dossier,
):  # à partir d'un dossier de plusieurs run, on obtient la liste de toutes les run qui y figure
    list_subfiles = [p for p in Path(nom_du_dossier).iterdir()]

    return list_subfiles


def get_all_runs(
    list_subfiles,
):  # une fois la liste des sous dossiers extraite, cette fonction extrait les logger et meta données de toutes les runs
    List_logger = []
    List_meta = []
    for path in list_subfiles:
        with open(path / "metrics.runtime.json", "r") as f:
            data = json.load(f)
        List_logger.append(data)
        with open(path / "meta.json", "r") as g:
            meta = json.load(g)
        List_meta.append(meta)
    return List_logger, List_meta


def determiner_variable(
    List_logger, List_meta
):  # détermine quel unique paramètre varie entre les logger
    liste_meta = list(List_meta[0].keys())  # list des différentes meta données
    for param in liste_meta:
        if (
            List_meta[0][param] != List_meta[1][param]
        ):  # on cherche la metadonnée qui varie entre les 2 premiers runs
            param_variable = param
    return liste_meta, param_variable


def sort(
    List_logger, List_meta, param_variable
):  # cette fonction assure que les loggers sont triés selon le paramètre d'intérêt
    # 1. On "colle" les deux listes ensemble (paires [logger, meta])
    paires = zip(List_logger, List_meta)

    # 2. On trie les paires en regardant la métadonnée (élément d'index 1 du tuple)
    # Exemple : on trie par 'L' croissant
    paires_triees = sorted(paires, key=lambda paire: paire[1][param_variable])

    # 3. On "décolle" (unzip) pour retrouver nos deux listes séparées mais triées
    # L'opérateur * permet de dépaqueter
    List_logger, List_meta = zip(*paires_triees)

    # Note : zip renvoie des tuples, on convertit en liste si besoin
    List_logger = list(List_logger)
    List_meta = list(List_meta)
    return List_logger, List_meta


def generer_titre(
    List_meta, param_variable
):  # génère le titre qui va bien avec tous les pramètres fixes
    """
    Génère un titre à partir des métadonnées communes, en excluant la variable.

    Args:
        List_meta (list): La liste de dictionnaires de métadonnées.
        param_variable (str): Le nom de la clé à exclure (ex: 'lr', 'n'...).
    """

    # 2. On prend les métadonnées du premier run comme référence
    # (On suppose que les autres runs ont les mêmes paramètres fixes)
    meta_ref = List_meta[0]

    elements_titre = []

    # 3. On parcourt les clés et on filtre
    # On peut trier les clés (sorted) pour que l'ordre soit toujours le même
    for key, value in meta_ref.items():
        # CONDITION D'EXCLUSION
        # On ignore la variable demandée ET les clés "techniques" (optionnel)
        if key != param_variable and key != "raw_name":
            # On formate joliment (ex: "L=4")
            elements_titre.append(f"{key}={value}")

    # 4. On assemble le tout avec un séparateur (ex: " | " ou " - ")
    titre_final = " | ".join(elements_titre)

    return titre_final


def afficher(List_logger, List_meta):
    liste_meta, param_variable = determiner_variable(List_logger, List_meta)
    e_gs = List_meta[0]["exact"]
    colors = cm.magma(np.linspace(0, 1, len(List_logger)))
    plt.figure(figsize=(10, 6))
    for i in range(len(List_logger)):
        plt.subplot(1, 2, 1)
        plt.plot(
            List_logger[i]["Energy"]["iters"],
            List_logger[i]["Energy"]["Mean"],
            color=colors[i],
            label=f"{param_variable}={List_meta[i][param_variable]}",
        )
        plt.subplot(1, 2, 2)
        plt.semilogy(
            List_logger[i]["Energy"]["iters"],
            np.abs(np.array(List_logger[i]["Energy"]["Mean"]) - e_gs) / np.abs(e_gs),
            color=colors[i],
            label=f"{param_variable}={List_meta[i][param_variable]}",
        )
    # Ligne horizontale pour l'énergie exacte
    plt.subplot(1, 2, 1)
    plt.axhline(y=e_gs, color="red", linestyle="--", label="Exact ground state")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    titre = (
        "VMC_Energy_vs_Iteration_for_different_"
        + param_variable
        + ",_"
        + generer_titre(List_meta, param_variable)
    )
    print(titre)
    titre = "\n".join(textwrap.wrap(titre, width=50))
    plt.title(titre, fontsize=10)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\frac{|E_{MC}- E_{GS}|}{E_{GS}}$")
    plt.title("Error vs Iteration (log scale)")
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot(file_path):  # de la forme r'chemin\du\dossier'
    list_subfiles = get_sub_files(file_path)
    List_logger, List_meta = get_all_runs(list_subfiles)
    liste_meta, param_variable = determiner_variable(List_logger, List_meta)
    List_logger, List_meta = sort(List_logger, List_meta, param_variable)
    afficher(List_logger, List_meta)


plot(
    r"C:\Users\mouts\OneDrive\Bureau\X\2A\PSC\NQS\runs\alpha"
)  # à titre d'exemple, on plot tous les runs du dossier 'run'
