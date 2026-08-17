# Machine Learning Demos

Static GitHub Pages site for course-native interactive machine-learning demonstrations.

## Local structure

- `index.html` — homepage and demo directory
- `backpropagation-xor/` — classroom version: fixed 2–2–1 network with sigmoid activation
- `backpropagation-xor-explorer/` — experimental configurable version
- `autoencoder/` — 10×10 bitmap autoencoder with selectable datasets, hidden width, augmentation, feature maps, and reconstruction diagnostics
- `random-forest/` — random-forest voting lab showing bootstrap samples, random feature splits, and majority voting
- `xgboost/` — round-by-round multi-class gradient boosting lab (2 features, 2–3 classes, one tree per class per round)
- `parzen-knn/` — Parzen window vs. k-NN classification lab with a leave-one-out accuracy curve for window/k selection
- `bayes-fish-demo/`, `bayes-rain-demo/`, `pens-confusion-matrix-demo/`, `sampling-quantization-demo/` — ported from the sibling [`image-processing-demos`](https://github.com/punpiti/image-processing-demos) repo (Bayes classification/theorem, confusion-matrix metrics, and sampling/quantization); each is a self-contained single-file page with its own design system, not the shared `shared.css`/`shared-layout.js` chrome used elsewhere in this repo

To publish with GitHub Pages, put this directory in a repository named `machine-learning-demos`, then enable Pages from the repository's default branch and root directory.
