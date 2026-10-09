# TP 2 — Régression linéaire (M2 IWOCS, Apprentissage Automatique)

**Hamadou BA**

| Fichier | Contenu |
|---|---|
| `TP_Regression_lineaire_Hamadou_BA.ipynb` | Notebook de rendu : énoncé officiel complet + code + réponse rédigée sous chaque question |
| `TP_Regression_lineaire_Hamadou_BA.pdf` | Export PDF du notebook de rendu |
| `exercice1.ipynb`, `exercice2.ipynb` | Les mêmes réponses, séparées par exercice |
| `ex1x.dat`, `ex1y.dat`, `ex2x.dat`, `ex2y.dat` | Données fournies |

Lancer : `pip install numpy matplotlib scikit-learn jupyter` puis `jupyter notebook`.
Les notebooks sont livrés déjà exécutés.

## Principe commun (cours, chapitre 6)

- Fonction de coût (moindres carrés) : `J(θ) = 1/(2m) · Σ (h_θ(xᵢ) − yᵢ)²`
- Descente de gradient : `θ* = θ − α · ∇J(θ)`, avec mise à jour simultanée de toutes les composantes.
- Critère d'arrêt imposé par l'énoncé : `|J(θ*) − J(θ)| / J(θ) < 10⁻³`.

## Exercice 1 — Taille d'un enfant selon son âge (50 enfants)

Modèle `h(x) = θ₀ + θ₁·x`, α = 0,07, θ = (0, 0) au départ.

- Q6 : après 5 itérations, la droite n'est pas ajustée et θ₁ oscille (≈ 0,38 / ≈ 0,01).
- Q7 : convergence en **407 itérations**, θ ≈ (0,714 ; 0,0705). Un tableau compare avec ε = 10⁻⁶, ε = 10⁻¹⁰ et la solution exacte (0,7502 ; 0,0639).
- Q8 : **0,925 m** (3 ans), **1,066 m** (5 ans), **1,207 m** (7 ans).
- Q9 : surface 3D et courbes de niveau. J est une cuvette convexe avec un minimum unique, et sa vallée très allongée explique la convergence lente.

## Exercice 2 — Prix d'un logement à Portland (47 logements)

Modèle `h(X) = Xθ`, avec X augmentée d'une colonne de 1 et normalisée par `StandardScaler`.

- Q3 : `E = Xθ − Y`, `J = EᵗE / 2m`, `θ* = θ − α/m · XᵗE`.
- Q4 : avec α = 0,07, convergence en 67 itérations, θ ≈ (337 780 ; 102 295 ; 531).
- Q5 : 9 valeurs de α entre 0,001 et 10, J tracé sur 50 itérations.
  - α ≤ 0,01 est trop lent.
  - α = 0,3 et α = 1 sont les plus rapides.
  - α ≥ 3 diverge.
  - On retient **α = 1** : convergence en 8 itérations, θ ≈ (340 413 ; 108 390 ; −6 515), proche de la solution exacte (340 413 ; 109 448 ; −6 578).
- Prédiction pour 1650 m2 et 3 pièces, avec le point normalisé par le même `scaler.transform` : **≈ 293 500** (293 081 avec la solution exacte).
