# Plant Disease Classification Under Domain Shift: A Trustworthy AI Study

## Overview

Most plant disease classifiers report impressive accuracy — but that accuracy is almost always measured on clean, lab-controlled images. This project asks a different question: **what happens when a model trained in the lab is deployed in the real world?**

Using a ResNet18 classifier trained on the PlantVillage dataset, this project measures the accuracy drop when tested on real-world field photos (PlantDoc), investigates *why* some diseases survive this shift better than others, and builds a system that can recognize its own unreliability — through uncertainty estimation, out-of-distribution detection, and visual explainability — rather than failing silently.

## Motivation

Plant disease detection is a genuinely useful real-world application of computer vision — farmers and agricultural workers could use a phone camera to get an instant diagnosis. But nearly all public datasets and benchmark models are trained and evaluated on clean, isolated, lab-photographed leaves. A model that hits 90%+ accuracy in this setting tells you very little about how it will perform on a blurry, cluttered, naturally-lit photo taken in an actual field — which is exactly the setting the tool would need to work in to be useful.

This project treats that gap as the central research question, rather than an inconvenient footnote.

## Approach

**1. Baseline classifier** — Transfer learning with a frozen, ImageNet-pretrained ResNet18 backbone and a custom trainable final layer, trained on 15 disease classes across Pepper, Potato, and Tomato (PlantVillage dataset).

**2. Domain-shift evaluation** — The trained model (no retraining) is evaluated on PlantDoc, a dataset of real-world field photographs, using a manually verified class mapping between the two datasets' differing label schemes.

**3. Uncertainty estimation (MC Dropout)** — A Dropout layer is added before the final classification layer. At inference, each image is passed through the model ~30 times with dropout active, and the consistency of the predictions is used as a confidence score — testing whether the model "knows when it doesn't know."

**4. Out-of-distribution (OOD) detection** — The model is tested on images of plant species it was never trained on (Corn, Apple, Grape), to check whether its confidence score correctly drops on genuinely unfamiliar inputs, not just on familiar-but-difficult ones.

**5. Explainability (Grad-CAM)** — Gradient-based class activation maps are generated to visualize which regions of an image the model relies on for its predictions, used to sanity-check both correct and incorrect predictions.

## Key Findings

- **Baseline accuracy:** ~90% validation accuracy on PlantVillage's clean lab-style images.
- **Domain-shift collapse:** Accuracy dropped to ~25% overall on PlantDoc's real-world field images — a severe, but not uniform, decline.
- **Lesion size matters:** Diseases with large, high-contrast lesions (e.g. Tomato Late Blight, ~69% on PlantDoc) held up far better under domain shift than diseases with small, subtle symptoms (e.g. Tomato Bacterial Spot, <1% on PlantDoc). The likely cause: resizing field images down to the model's 224×224 input size shrinks small lesions to near-invisibility, while large lesions remain visible.
- **Uncertainty is a real but imperfect signal:** MC Dropout confidence averaged 0.72 on correct predictions vs. 0.65 on wrong ones. The gap was clearest at the extremes — very high confidence (0.9+) reliably meant a correct prediction, and very low confidence reliably meant a wrong one — but the signal was noisier in the middle range.
- **OOD detection partially works:** Confidence on genuinely unfamiliar species (Corn: 0.57, Apple: 0.57) was lower than even the model's average confidence on ordinary in-distribution mistakes (0.65) — a good sign. However, this didn't hold for all species tested (Grape: 0.64, indistinguishable from ordinary errors), showing the method isn't fully reliable.
- **Explainability reveals a likely root cause:** Grad-CAM on a correct, confident prediction showed the model's attention correctly focused on the leaf and lesion area. On a wrong prediction (a real PlantDoc image), the heatmap was instead concentrated almost entirely on irrelevant background clutter — visual evidence that a model trained only on clean, isolated-leaf images never learned to ignore background noise, because during training there essentially wasn't any.

## Tech Stack

- **PyTorch** / **torchvision** — model, training, transfer learning
- **pytorch-grad-cam** — explainability
- **Google Colab** (GPU runtime) — development environment
- **Kaggle API** — dataset access
- **Datasets:** [PlantVillage](https://www.kaggle.com/datasets/adilmubashirchaudhry/plant-village-dataset), [PlantDoc](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset)

## Limitations

- **Scope:** Limited to 3 plant species (Pepper, Potato, Tomato) and 15 disease classes, not the full range of species PlantVillage covers.
- **Data quality:** PlantDoc's "healthy tomato leaf" class was found to contain inconsistent data — including stock illustrations and whole-plant/seedling photos rather than clean single-leaf images — and its near-0% accuracy on this class likely reflects that data quality issue rather than a genuine domain-shift failure. This class was noted separately from genuine findings rather than folded into headline results.
- **OOD detection inconsistency:** Confidence-based OOD detection worked clearly for some unfamiliar species (Corn, Apple) but not others (Grape), suggesting the method is a useful but not fully reliable signal on its own.
- **Sample sizes:** Some evaluations (e.g., MC Dropout confidence comparison) were run on a random sample of PlantDoc images rather than the full dataset, for practicality.

## Project Structure

```
├── notebook.ipynb          # Full training, evaluation, and analysis pipeline
├── README.md                # This file
```

## Future Work

- Extend to the full PlantVillage species range
- Explore data augmentation or domain-adaptation techniques to close the domain-shift gap directly
- Test additional OOD detection methods (e.g., Mahalanobis distance, energy-based scoring) for more consistent results
