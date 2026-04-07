# Fabula-fr

`fabula-fr` est une bibliothèque Python pour analyser l’évolution du sentiment et/ou des émotions dans un texte et produire des **arcs narratifs** lissés. 

Le package fournit :

- une API Python (`Fabula`),
- une CLI (`fabula score` / `fabula arc`),
- des utilitaires de segmentation,
- des fonctions de resampling + lissage,
- des helpers de visualisation.

---

## Installation

```bash
pip install fabula-fr
```

Pour le développement local :

```bash
pip install -e .[dev]
```

---

## Démarrage rapide

### API Python

```python
from fabula.core import Fabula
from fabula.scorer import TransformersScorer

scorer = TransformersScorer(model="cmarkea/distilcamembert-base-sentiment")
fb = Fabula(scorer=scorer, analysis="sentiment")

text = "Bonjour. C'est une belle journée. Je suis triste."

# Scores par segment
scores = fb.score(text)
print(scores[["idx", "rel_pos", "label", "score"]])

# Arc narratif lissé
arc = fb.arc(text, n_points=100, smooth_window=7)
print(len(arc.x), len(arc.y))

```

### CLI

```bash
# Scoring par segments (JSON)
fabula score mon_texte.txt --format json

# Scoring par segments (CSV)
fabula score mon_texte.txt --format csv

# Arc narratif
fabula arc mon_texte.txt --n-points 120 --smooth-window 9 --format csv
```

Lecture depuis stdin :

```bash
cat mon_texte.txt | fabula score - --format json
```

---

## Segmentation

### Segmenters disponibles

- `RegexSentenceSegmenter` : segmentation par ponctuation terminale.
- `ParagraphSegmenter` : segmentation par paragraphes (`\n\n`).
- `SlidingWindowTokenSegmenter` : fenêtres glissantes sur tokens.
- `DocumentChunkTokenSegmenter` : chunks de document par tokens.

### Choix via CLI

Actuellement la CLI expose :

- `--segment sentence`
- `--segment paragraph`

---

## Scoring et modèles

### Mode standard

Par défaut, la CLI utilise `TransformersScorer` avec le modèle :

- `cmarkea/distilcamembert-base-sentiment`

Vous pouvez le changer :

```bash
fabula score mon_texte.txt --model astrosbd/french_emotion_camembert
```

### Mode sans téléchargement (`--dummy`)

Pour des tests rapides sans dépendre de Hugging Face :

```bash
fabula score mon_texte.txt --dummy
fabula arc mon_texte.txt --dummy
```

---

## Arcs narratifs

`Fabula.arc(...)` :

1. récupère les scores segmentés,
2. interpole vers `n_points` (`resample_to_n`),
3. applique un lissage (`smooth_series`).

Vous pouvez aussi générer des arcs multi-séries avec `score_col="probs"` dans l’API Python.

---

## Visualisation

Le module `fabula.plot` expose :

- `plot_arc(...)` pour un arc scalaire,
- `plot_arc_series(...)` pour plusieurs séries.

Exemple :

```python
from fabula.plot import plot_arc

fig, ax = plot_arc(arc, title="Arc narratif", save_path="arc.png")
```

---

## Tests

Exécuter les tests :

```bash
python -m pytest -q
```

## Citation

Si vous utilisez **fabula**, merci de citer :

Cafiero, Florian, et Alexandre Lionnet.  *Des émotions au fil du récit : fabula, un package pour analyser les textes francophones par Transformers*. Communication présentée à Humanistica 2026 (Paris), à paraître dans *Anthology of Computers and the Humanities*.
