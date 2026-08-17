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
- `sgd-vs-lda/` — step-by-step SGD training of a linear classifier (perceptron/step or logistic/sigmoid activation, selectable) on a selectable rain dataset (overlapping/separable/imbalanced), with a live calculation trace and comparison against LDA's closed-form boundary
- `svm-margin/` — step-by-step soft-margin SVM training (Pegasos-style hinge-loss update) on the same three rain datasets as `sgd-vs-lda/`, with a λ slider controlling margin width, support vectors outlined live on the scatter, and the same LDA comparison
- `bayes-fish-demo/`, `bayes-rain-demo/`, `pens-confusion-matrix-demo/`, `sampling-quantization-demo/` — ported from the sibling [`image-processing-demos`](https://github.com/punpiti/image-processing-demos) repo (Bayes classification/theorem, confusion-matrix metrics, and sampling/quantization); each is a self-contained single-file page with its own design system, not the shared `shared.css`/`shared-layout.js` chrome used elsewhere in this repo
- `data-to-decision/` — no-formulas lab for why a decision needs data: same 48 days, evidence columns (season, cloud, humidity) revealed one at a time, accuracy climbing at each stage
- `model-selection/` — train/validation/test workflow on the Iris dataset: fit three named model families on train, pick a winner on validation, then open the test set once
- `activation-functions/` — sigmoid/tanh/arctan/ReLU/step/linear all train simultaneously on a selectable rain dataset (same three options as `sgd-vs-lda/`), same steepness k and learning rate, error-vs-epoch overlaid on one chart, to see why a step function's zero slope freezes gradient learning while steep smooth activations oscillate or saturate — and how much of that depends on the data itself

The homepage groups demos by chapter, following `~/OneDrive/book/machine-learning`'s `MASTER_OUTLINE.md` rather than either course's own week/session numbering.

To publish with GitHub Pages, put this directory in a repository named `machine-learning-demos`, then enable Pages from the repository's default branch and root directory.
