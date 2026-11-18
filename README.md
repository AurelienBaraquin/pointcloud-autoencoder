# pointcloud-autoencoder

Un autoencodeur de nuages de points en PyTorch, avec un encodeur de type PointNet et une perte de Chamfer.

**Question :** un autoencodeur entraîné uniquement sur des formes pleines sait-il reconstruire des pièces minces, comme de la tôle emboutie ? Et comment l'améliorer ?

Toutes les données sont générées par le code, rien n'est téléchargé :
- **formes pleines** : cubes, cylindres ;
- **pièces minces** : plaques pliées, coques courbes.

## Installation

Python 3.10 ou plus récent.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements.txt` contient ce qu'il faut pour exécuter le code, et `requirements-dev.txt` y ajoute les outils de vérification :

```bash
ruff check .   # analyse statique du code
pytest         # tests
```

## Arborescence prévue

```
src/
  shapes.py     # génération des nuages de points
  model.py      # encodeur PointNet + décodeur
  losses.py     # distance de Chamfer
  train.py      # boucle d'entraînement
  evaluate.py   # métriques et figures
tests/
results/        # courbes, tableaux, rendus
```

## Plan

### 1. Données (`shapes.py`)
- Chaque forme donne un tableau `(N, 3)`, avec par exemple N = 1024 ou 2048 points.
- **Formes pleines** : échantillonner la **surface** du cube et du cylindre, pas le volume. Chaque face reçoit un nombre de points proportionnel à son aire. Les dimensions varient au hasard.
- **Pièces minces** :
  - Une plaque pliée, c'est deux rectangles reliés par une arête, avec un angle de pli aléatoire.
  - Une coque courbe, c'est une portion de cylindre ou de sphère.
  - Elles ont une épaisseur `e`, par exemple 1 à 3 % de la taille. Ce sont les deux faces proches qui piègent le modèle.
- **Normalisation** : centrer chaque nuage et le ramener dans la sphère unité.
- Rotation aléatoire, et graine fixe (`seed`) pour pouvoir reproduire les résultats.

### 2. Modèle (`model.py`)
- **Encodeur PointNet** : un MLP partagé appliqué point par point (`Conv1d` 3→64→128→1024), puis un **max-pooling** global. On obtient un vecteur latent de taille 128 à 512. Le T-Net est optionnel.
- **Décodeur** : un MLP latent→256→512→N×3, puis un `reshape` en `(N, 3)`.

### 3. Perte de Chamfer (`losses.py`)
- `torch.cdist(pred, gt)` donne la matrice des distances, puis `min(dim=2).mean() + min(dim=1).mean()`.
- Choisir entre distances au carré ou non, et garder ce choix partout.
- Tests : deux nuages identiques donnent 0, et la perte est symétrique.

### 4. Expérience
1. **Référence** : entraîner sur les formes pleines seules. Évaluer ensuite sur un jeu de test pleines et sur un jeu de test minces, que le modèle n'a jamais vus. On s'attend à un Chamfer nettement plus élevé sur les pièces minces : les deux faces s'écrasent en une seule, et le pli devient arrondi.
2. **Améliorations**, à tester une par une :
   - A : entraîner sur un mélange pleines + minces ;
   - B : changer l'échantillonnage, par exemple en mettant plus de points près des arêtes et des plis ;
   - A + B.
3. **Mesure** : un tableau avec le Chamfer moyen par catégorie et par variante (moyenne ± écart-type sur 3 graines). Ajouter un rendu côte à côte entrée / reconstruction pour quelques pièces.

### 5. Pour aller plus loin
- **F-score à un seuil τ**, qui fait mieux ressortir le problème des parois fines que le Chamfer seul.
- Évaluer au même nombre de points partout, pour que l'échantillonnage B ne fausse pas la comparaison.

## Résultats

À venir.
