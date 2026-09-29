# TP Régression linéaire

Deux notebooks Jupyter : `exercice1.ipynb` (régression univariée) et `exercice2.ipynb` (régression multivariée, écriture matricielle).
Les données (`ex1x.dat`, `ex1y.dat`, `ex2x.dat`, `ex2y.dat`) sont dans le même dossier.

Lancer : `pip install numpy matplotlib scikit-learn jupyter` puis `jupyter notebook`.

## Principe commun

On cherche des paramètres θ tels que l'hypothèse `h_θ(x)` soit au plus près de la cible `y`.

- **Fonction de coût** (moindres carrés) : `J(θ) = 1/(2m) · Σ (h_θ(xᵢ) − yᵢ)²`
- **Descente de gradient** : à chaque itération, `θ ← θ − α · ∇J(θ)`, où α (taux d'apprentissage) règle la taille du pas.
  Trop petit : convergence lente ; trop grand : divergence.
- **Critère d'arrêt** : on s'arrête quand `|J(θ*) − J(θ)| / J(θ) < ε`. Ici **ε = 10⁻¹⁰** (variable `EPS` dans chaque notebook), pour que θ soit très proche de l'optimum.

## Exercice 1 — Taille d'un enfant en fonction de son âge

50 enfants de 2 à 8 ans. Modèle : `h(x) = θ₀ + θ₁·x` (une droite).

| Étape (question) | Ce que fait le notebook |
|---|---|
| 1–2 | Lit `x` (âge) et `y` (taille en m) avec `np.loadtxt`, trace le nuage de points. |
| 3 | Fonction `h(theta0, theta1, x)`. |
| 4 | Fonction `J(theta0, theta1)`. |
| 5 | Fonction `iteration` : `θ₀ ← θ₀ − α/m·Σ(h−y)` et `θ₁ ← θ₁ − α/m·Σ(h−y)·x`, mises à jour simultanées. α = 0,07, θ = (0, 0) au départ. |
| 6 | Quelques itérations (5) et tracé de la droite : elle n'est pas encore ajustée. |
| 7 | Boucle jusqu'au critère d'arrêt : 1512 itérations, θ₀ ≈ 0,750 et θ₁ ≈ 0,0639. La droite passe bien au milieu du nuage. |
| 8 | Prédictions : 3 ans → 0,942 m ; 5 ans → 1,070 m ; 7 ans → 1,197 m. |
| 9 | Calcule J sur une grille 100×100 (θ₀ ∈ [−30, 30], θ₁ ∈ [−3, 3]) et trace la surface 3D et les courbes de niveau. C'est une « cuvette » avec un seul minimum, ce qui explique que la descente de gradient converge. |

## Exercice 2 — Prix d'un logement (surface, nombre de pièces)

47 logements de Portland. Modèle : `h(X) = θ₀ + θ₁·surface + θ₂·pièces`.

| Étape (question) | Ce que fait le notebook |
|---|---|
| 1 | Lit `x` (47×2) et `y` (prix). |
| 2 | Normalise avec `StandardScaler` (moyenne 0, écart-type 1). Nécessaire car la surface (~2000) et le nombre de pièces (~3) n'ont pas le même ordre de grandeur, sinon la descente converge très mal. |
| 3 | Version matricielle : on ajoute une colonne de 1 à X ; `h = Xθ` ; `E = Xθ − Y` ; `J = EᵀE / 2m` ; itération `θ ← θ − α/m · Xᵀ E`. |
| 4 | Régression avec α = 0,07 jusqu'à convergence (320 itérations). |
| 5 | Teste 9 valeurs de α entre 0,001 et 10 sur 50 itérations et trace J en fonction du nombre d'itérations (échelle log). α ≤ 0,03 : lent ; α = 0,3 et 1 : très rapide ; α ≥ 3 : divergence. On retient **α = 1**, on recalcule θ jusqu'à convergence (22 itérations) et on obtient θ ≈ (340413 ; 109447 ; −6578), identique à la solution exacte (équations normales) à 4 décimales près, affichée pour vérification. |
| Prédiction | Logement de 1650 m² et 3 pièces : le point est **normalisé avec le même scaler** puis passé dans `h` : environ **293 000**. |

## Points d'attention

- Toujours appliquer à un nouveau point la même normalisation que celle des données d'entraînement (`scaler.transform`, pas `fit_transform`).
- Les notebooks sont livrés déjà exécutés, avec leurs sorties et leurs graphiques.
