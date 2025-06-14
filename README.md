# [Topological Analysis of Airbnb Price Dynamics in Cartagena](https://topological-analysis-of-airbnb-price.onrender.com/)

> 🌐 Live Dashboard: [https://topological-analysis-of-airbnb-price.onrender.com/](https://topological-analysis-of-airbnb-price.onrender.com/)

---

## Overview

This project presents an interactive Dash dashboard and accompanying analysis pipeline for exploring, segmenting, and predicting short-term rental prices on Airbnb in Cartagena. We combine classical statistical methods, unsupervised learning, and **Topological Data Analysis (TDA)** to:

* Investigate distributions and volatility of nightly base fees.
* Segment listings via **PCA + K-Means**, hierarchical clustering, volatility-based clustering and amenity-based clustering.
* Uncover intrinsic “shape” of the data using **UMAP embeddings**, **Vietoris–Rips complexes**, **persistent homology**, and **Betti numbers**.
* Build and evaluate classification models (XGBoost, Random Forest + L1-Logistic, Gradient Boosting + Decision Tree) to predict price tiers, Superhost status, and amenity-based clusters.
* Propose a feature-selection and dimensionality-reduction strategy combining **PCA**, **UMAP**, and topological summaries.

Potential applications include dynamic pricing optimization, anomaly detection, host segmentation, and robust feature engineering for recommendation systems.

---

## Key Concepts

* **Data Cleaning & Interpolation:**

  * Handling missing values in daily price series via K-Nearest Neighbors (KNN) imputation and forward/backward fill.
* **Dimensionality Reduction:**

  * **Principal Component Analysis (PCA):** Linear projection maximizing variance.
  * **Uniform Manifold Approximation and Projection (UMAP):** Nonlinear embedding preserving local/global structure in 2D/3D.
* **Clustering Methods:**

  * **K-Means:** Partitioning into *k* centroids via Euclidean distance.
  * **Agglomerative Clustering (Ward linkage):** Hierarchical clustering minimizing variance.
  * **Volatility Clustering:** K-Means on derived features—relative standard deviation, maximum deviation, spike frequency.
* **Topological Data Analysis (TDA):**

  * **Vietoris–Rips Complex:** Simplicial complex built over point cloud at varying distance thresholds.
  * **Persistent Homology:** Computation of homology groups $H_0$, $H_1$, $H_2$ across filtrations, birth–death pairs, barcodes, and persistence diagrams.
  * **Betti Numbers ($\beta_0,\beta_1,\beta_2$):** Quantify connected components, cycles, and voids.
  * **Bifiltration & Bipersistence Heatmap:** Two-parameter analysis of $\beta_0(p, v)$ over price mean vs. volatility.
  * **Sliding-Window Persistence:** Time-varying topological features over price windows.
* **Supervised Learning & Model Evaluation:**

  * **XGBoost:** Gradient-boosted decision trees for multiclass classification (price tiers).
  * **Voting Ensembles:** Soft-voting combination of Random Forest, Logistic Regression (L1 penalty), Gradient Boosting, and Decision Trees.
  * **Cross-Validation & Grid Search:** Stratified K-fold, hyperparameter tuning via `GridSearchCV`.
  * **Metrics:** Accuracy, precision, recall, F1-score, ROC AUC, Precision–Recall curves, confusion matrices.
  * **Feature Importance:** Extraction from tree-based models to rank variables.

---

## Dashboard Architecture

* **Front-end:** Dash (Plotly) with Bootstrap Darkly theme; interactive tabs for EDA, Clustering, TDA, and Classification.
* **Data Collection:**

  * Airbnb listings were scraped using [Apify's Airbnb Listings Scraper](https://console.apify.com/actors/OIYrZy1OpUEgIMYmh/input), enabling bulk download of metadata (availability, pricing, ratings, and amenities) for each property.
* **Data Storage:** SQLite database containing three tables—`Attributes`, `TimeSeriesRaw`, and `TimeSeriesInterpolated`.
* **Data Processing:** Python (pandas, NumPy, SciPy) for ETL and feature engineering.
* **Visualization:** Plotly Express & Graph Objects for histograms, scatterplots, 3D embeddings, dendrograms, heatmaps, bar/box charts, and topological complexes.
* **Topology:** GUDHI library for simplex tree construction and persistence computation.
* **Embedding & Clustering:** Scikit-learn’s `StandardScaler`, `PCA`, `KMeans`, `AgglomerativeClustering`, UMAP.

---

## Deployment

* The dashboard is deployed on [Render](https://render.com/), allowing public access to all visualizations and results via the following link:
  👉 **[https://topological-analysis-of-airbnb-price.onrender.com/](https://topological-analysis-of-airbnb-price.onrender.com/)**

---

## Dependencies

* **Core Python Libraries:**

  * `pandas`, `numpy`, `scipy` (statistical functions, KDE, distance metrics)
  * `scikit-learn` (scaling, PCA, clustering, regression, model selection, metrics)
  * `umap-learn` (UMAP embedding)
  * `gudhi` (TDA: Vietoris–Rips, persistence)
* **Visualization & Dashboard:**

  * `plotly` (Express & Graph Objects)
  * `dash`, `dash-bootstrap-components` (interactive web app)
  * `dash-table` (data tables)
* **Database & Utils:**

  * `sqlite3` (data storage)
  * `re`, `random`, `warnings` (utility functions)
* **Machine Learning Models:**

  * `XGBoost`
  * `Random Forest`
  * `Logistic Regression (L1 penalized)`
  * `Voting Classifier (soft voting)`
  * `Decision Tree`
  * `Gradient Boosting`
* **Other:**

  * `statsmodels` or `scipy.stats` (OLS trendlines, KDE)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bacaSantiago/Topological-Analysis-of-Airbnb-Price-Dynamics-in-Cartagena
   cd airbnb-cartagena-tda
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare the data:**

   * Place `airbnb_cartagena.sqlite` in the project root (contains Attributes, TimeSeriesRaw, TimeSeriesInterpolated).
   * Ensure asset files (GIFs, JSON metrics) are in `assets/` and `models/`.

4. **Run the dashboard:**

   ```bash
   dashboard app.py
   ```

   Navigate to `http://127.0.0.1:8050` in your browser.

---

## Potential Applications

* **Host Pricing Optimization:** Segment properties and identify pricing anomalies for dynamic rate adjustments.
* **Market Analysis:** Discover latent sub-markets via topology (cycles and components) for targeted promotions.
* **Anomaly Detection:** Flag “disconnected” listings with atypical pricing behavior.
* **Feature Engineering:** Leverage TDA summaries (sum of lifetimes, Betti curves) as robust predictors in recommendation systems.
* **Generalizable Framework:** Adapt methodology to other P2P platforms, geographies, or time windows.
