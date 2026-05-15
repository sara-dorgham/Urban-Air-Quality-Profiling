# 🌍 Air Quality Analysis &  Clustering Project

## 📌 Project Overview
This project provides a comprehensive end-to-end Machine Learning pipeline to analyze, preprocess, and cluster air quality data. Using sensors' chemical responses, we categorize air quality into distinct levels, identifying pollution patterns and their correlation with human activity (Rush Hours) and environmental factors.

---

## 👥 Team Contributions & Responsibilities

| Member | Role | Key Achievements |
| :--- | :--- | :--- |
| **Malak** | EDA & Preprocessing | Data cleaning, handling -200 null markers, and outlier clipping. |
| **Abdelrahman** | Visualization | Designed comparative plots (Before/After) and rush-hour trends. |
| **Abdalla** | Feature Engineering | Developed PCA components and circular time-based features. |
| **Sara** | Model Development | Implemented K-Means (Elbow Method) and optimized DBSCAN. |
| **Mahmoud** | GUI Development | Built an interactive interface for real-time data exploration. |

---

## 🛠 Detailed Technical Pipeline

### 1. Exploratory Data Analysis (EDA)
In this phase, we performed a deep dive into the dataset's structure:
* **Missing Value Identification:** Discovered that `-200` was used as a placeholder for missing values. These were converted to `NaN` for proper statistical handling.
* **Structural Cleaning:** Removed empty "Unnamed" columns and corrected data types for the `Date` and `Time` features.
* **Statistical Profiling:** Analyzed mean, variance, and distribution of sensors like `PT08.S1(CO)` and `NMHC(GT)`.

### 2. Advanced Preprocessing
Cleaning data is 80% of the work. We applied several techniques:
* **Time-Series Imputation:** Used Forward Fill (`ffill`) and Backward Fill (`bfill`) to handle gaps, ensuring continuity in time-dependent sensor data.
* **Specific Day Analysis:** Identified 2004-11-03 as an anomalous day with sensor malfunctions and removed the corrupted segments.
* **Outlier Mitigation (Clipping):** Instead of removing all outliers, we capped extreme values at the 1st and 99th percentiles to retain data volume while reducing noise.

### 3. Data Visualization Strategy
We utilized visualization to tell a story:
* **Before Cleaning:** Used `missingno` to highlight data sparsity and histograms to show the bias caused by invalid values.
* **After Cleaning:** Created **Heatmaps** to identify multicollinearity between sensors. 
* **Pollution Trends:** Developed line plots to visualize the **Rush Hour Effect**, showing clear spikes in CO and NOx during 8-9 AM and 6-8 PM.

### 4. High-Level Feature Engineering
To improve clustering performance, we created new informative features:
* **Atmospheric Interaction:** Created `T_RH_Interaction` (Temperature × Humidity) to capture how weather affects sensor sensitivity.
* **Dimensionality Reduction (PCA):** Fused 5 different sensor readings into two **Principal Components (PCA1 & PCA2)**, capturing over 90% of the variance while reducing model complexity.
* **Circular Encoding:** Applied Sine/Cosine transformations to the `Hour` feature, ensuring the model understands that hour 23 is close to hour 0.
* **Lag Features:** Added 1-hour lag and 3-hour rolling averages to capture pollution momentum.

### 5. Unsupervised Model Development
We compared two distinct clustering approaches:

#### A. K-Means Clustering
* **Optimization:** Used the **Elbow Method** (Inertia vs. K) to find the "K" point.
* **Evaluation:** Achieved a **Silhouette Score** to measure cluster separation and cohesion.
* **Interpretation:** Successfully segmented the data into 3 levels: Low, Moderate, and High Pollution zones.

#### B. DBSCAN (Density-Based Spatial Clustering)
* **Parameter Tuning:** Used a **K-Distance Graph** to mathematically determine the optimal `eps` value.
* **Noise Detection:** Effectively separated anomalous sensor readings (Noise) from the core data density.

### 6. Interactive GUI
Mahmoud developed a user-friendly interface that allows users to:
* Load new datasets dynamically.
* Visualize clustering results on the fly.
* Check specific pollution metrics for any given hour.

---

## ⚙️ Requirements & Installation
Ensure you have Python 3.8+ installed, then run:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn missingno