# Recadrage Intelligent d'Image au Format 9:16 par Homographie

> Script Python de recadrage et redressement de documents et écrans photographiés de biais, avec correction manuelle interactive des coins.

---

## Table des matières

- [Aperçu](#aperçu)
- [Principe mathématique](#principe-mathématique)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Options](#options)
- [Interface interactive](#interface-interactive)
- [Architecture du code](#architecture-du-code)
- [Exemples](#exemples)
- [Limitations connues](#limitations-connues)
- [Structure du projet](#structure-du-projet)
- [License](#license)

---

## Aperçu

Ce script permet de recadrer intelligemment une image contenant un document (feuille de papier, contrat, carte) ou un écran (moniteur, tablette, smartphone) photographié en conditions réelles — c'est-à-dire avec un angle de prise de vue non frontal — et de restituer une image redressée au format portrait **9:16** (1080 × 1920 px par défaut).

Le pipeline repose sur :

1. la **correction automatique de l'orientation** via les métadonnées EXIF ;
2. la **détection automatique du quadrilatère** dominant (contours + approximation polygonale) ;
3. une **interface de correction manuelle** par clic-glisser sur les coins détectés ;
4. le **calcul d'une homographie exacte** (`getPerspectiveTransform`) à partir de 4 paires de points ;
5. l'application de la **transformation de perspective** (`warpPerspective`) vers la résolution cible.

---

## Principe mathématique

Une homographie est une transformation projective représentée par une matrice **H** de taille 3×3, qui met en correspondance chaque point de l'image source avec un point de l'image destination dans l'espace projectif homogène :

```
x' = H · x      avec  x = (u, v, 1)ᵀ  et  x' = (u', v', 1)ᵀ
```

Ici, **H** est calculée à partir de **4 paires de points** (les coins du quadrilatère détecté dans la source, et les 4 coins du rectangle 9:16 en destination) via `cv2.getPerspectiveTransform`. Contrairement à `findHomography` qui requiert de nombreux points et utilise RANSAC, cette approche est **exacte et déterministe** dès lors que les 4 coins sont correctement positionnés.

---

## Prérequis

- Python **3.10** ou supérieur
- `opencv-contrib-python` (**obligatoire** — la version `opencv-python` seule ne supporte pas l'affichage de fenêtres sous Windows)

---

## Installation

**1. Cloner le dépôt**

```bash
git clone https://github.com/mwa1310/cv_homograhy.git
cd cv_homography
```

**2. Créer un environnement virtuel (recommandé)**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

> ⚠️ Si vous aviez déjà installé `opencv-python`, désinstallez-le d'abord :
> ```bash
> pip uninstall opencv-python opencv-python-headless
> pip install opencv-contrib-python
> ```

---

## Utilisation

```bash
python recadrage_avec_homographie.py <image_source> [options]
```

### Exemples rapides

```bash
# Document (feuille de papier) – mode interactif
python recadrage_avec_homographie.py contrat.jpg --mode document

# Écran (moniteur, tablette) – mode interactif
python recadrage_avec_homographie.py ecran.jpg --mode ecran

# Entièrement automatique, sans interface
python recadrage_avec_homographie.py photo.jpg --auto

# Résolution de sortie personnalisée
python recadrage_avec_homographie.py photo.jpg --width 720 --height 1280 --output resultat.jpg
```

---

## Options

| Option | Raccourci | Valeur par défaut | Description |
|---|---|---|---|
| `source` | — | *(obligatoire)* | Chemin vers l'image source |
| `--mode` | `-m` | `document` | Profil de détection : `document` ou `ecran` |
| `--output` | `-o` | `recadre_9_16.jpg` | Chemin de l'image de sortie |
| `--width` | — | `1080` | Largeur cible en pixels |
| `--height` | — | `1920` | Hauteur cible en pixels |
| `--auto` | — | `False` | Mode automatique sans interface de correction |

---

## Interface interactive

Lorsque le script est lancé sans `--auto`, une fenêtre OpenCV s'ouvre et affiche l'image avec les **4 coins détectés automatiquement**.

```
┌─────────────────────────────────────┐
│  HG ●───────────────────● HD        │
│     │                   │           │
│     │   Zone détectée   │           │
│     │                   │           │
│  BG ●───────────────────● BD        │
└─────────────────────────────────────┘
```

| Action | Effet |
|---|---|
| **Clic gauche + glisser** sur un coin | Repositionner ce coin |
| **Entrée** | Valider les coins et lancer le recadrage |
| **R** | Réinitialiser les coins à la détection automatique |
| **Q** ou **Échap** | Quitter sans sauvegarder |

---

## Architecture du code

```
recadrage_avec_homographie.py
│
├── correct_exif_orientation()   # Étape 1 – Correction orientation EXIF
│
├── preprocess_document()        # Étape 2 – Prétraitement image de document
├── preprocess_screen()          #         – Prétraitement image d'écran
│
├── order_corners()              # Étape 3 – Ordonnancement des 4 coins
├── detect_quad()                #         – Détection du quadrilatère dominant
├── fallback_corners()           #         – Coins par défaut si détection échoue
│
├── CornerEditor                 # Étape 4 – Interface interactive (classe)
│   ├── _draw()
│   ├── _mouse_callback()
│   └── run()
│
├── build_destination_corners()  # Étape 5 – Coins destination 9:16
└── warp_image()                 #         – Calcul H + warpPerspective
```

---

## Exemples

### Cas 1 — Document détecté automatiquement

```
[INFO] Image chargée : 3024×4032 px
[INFO] Orientation EXIF corrigée.
[INFO] Quadrilatère détecté automatiquement.
[INFO] Coins finaux :
       [[  84.  102.]
        [2941.   98.]
        [2938. 3930.]
        [  81. 3927.]]
[INFO] Matrice d'homographie :
       [[ 3.7e-01  1.2e-03  8.4e+01]
        [ 2.1e-04  4.8e-01  1.0e+02]
        [ 1.3e-07  5.9e-08  1.0e+00]]
[OK] Image recadrée sauvegardée : recadre_9_16.jpg
     Résolution : 1080×1920 px
```

### Cas 2 — Aucun quadrilatère détecté (coins par défaut)

```
[AVERT] Aucun quadrilatère détecté -> coins par défaut utilisés.
```
→ L'interface s'ouvre avec une marge de 10 % sur chaque bord. L'utilisateur repositionne manuellement les coins.

---

## Limitations connues

- La détection automatique est sensible aux **fonds chargés** ou aux images avec de nombreux bords concurrents. Dans ce cas, préférez le mode interactif pour ajuster manuellement les coins.
- Le mode `ecran` peut être mis en défaut en présence de **reflets spéculaires intenses** couvrant une large portion de l'écran.
- Le script traite **une image à la fois**. Pour un traitement par lot, il peut être intégré dans une boucle externe.

---

## Structure du projet

```
cv_homography/
│
├── recadrage_avec_homographie.py   # Script principal
├── README.md   # Documentation 
└── requirements.txt   # Dépendences nécesaires

```

---

## License

Ce projet est sous license MIT. Voir [MIT License](LICENSE).

---
