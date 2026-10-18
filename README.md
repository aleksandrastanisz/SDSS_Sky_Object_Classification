# SDSS-17 Sky Object Classification & Redshift Analysis

Machine learning pipeline classifying celestial objects (Stars, Galaxies, Quasars) from Sloan Digital Sky Survey Data Release 17 (SDSS-17) using Random Forest. The project includes an ablation study evaluating the impact of spectroscopic redshift ($z$) versus broad-band photometry ($u, g, r, i, z$).

## Overview & Scientific Motivation
Broad-band optical colors of high-redshift quasars (QSO) often overlap with Galactic stars (such as white dwarfs and blue horizontal-branch stars). This project investigates whether 5-band SDSS photometry alone provides sufficient feature separation or if spectroscopic redshift measurement remains indispensable.

## Dataset & Features
The dataset consists of 100 000 observations partitioned into 80% training and 20% stratified test sets.
Input feature vectors:
- Photometric bands: $u$ (355 nm), $g$ (469 nm), $r$ (617 nm), $i$ (748 nm), $z$ (893 nm)
- Physical metric: spectroscopic redshift $z = (\lambda_{\text{obs}} - \lambda_{\text{emit}}) / \lambda_{\text{emit}}$

## Results & Ablation Study
Full feature set ($u, g, r, i, z$ + redshift) achieves **98% overall accuracy**:
- GALAXY: Precision 0.98, Recall 0.99, F1 0.98
- QSO: Precision 0.97, Recall 0.93, F1 0.95
- STAR: Precision 1.00, Recall 1.00, F1 1.00

Removing the redshift feature causes a massive drop in classifier discriminative power: star-galaxy confusion jumps to 669 errors, QSO-galaxy to 440, and galaxy-star to 369. Feature importance analysis shows redshift accounts for over 64% of node impurity reduction across the forest.

## Full Report
A complete technical report in Polish is available in this repository (`SDSS-17_Redshift.docx`).

## Setup & Run
```bash
pip install pandas scikit-learn matplotlib
python main.py
