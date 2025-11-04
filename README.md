# 🎬 Netflix Data Labeling Lab
> **End-to-End Weak Supervision, Data Augmentation, and Model Development Pipeline**

---

## 🧠 **Overview**
This project demonstrates a complete **data labeling and machine learning pipeline** using the **Netflix Titles dataset**.  
The goal is to automatically classify Netflix content as **family-friendly** or **not family-friendly** using **weak supervision** (rule-based labeling),  
followed by **data augmentation**, **data slicing**, and **model development** for robust, explainable results.

The lab mirrors real-world MLOps workflows — moving from raw data → weak labels → augmented dataset → trained models.

---

## ⚙️ **Pipeline Summary**
| Stage | Notebook | Description |
|:------|:----------|:-------------|
| 🟩 **Data Labeling** | `01_Data_Labeling.ipynb` | Generate weak labels using rule-based Labeling Functions (LFs). |
| 🟨 **Data Augmentation** | `02_Data_Augmentation.ipynb` | Expand and balance the dataset with text-based augmentations. |
| 🟦 **Data Slicing & Model Development** | `03_Model_Development.ipynb` | Train models, evaluate across slices, and save deployable artifacts. |

---

## 🗂️ **Project Structure**

netflix_lab/
├── data/

│   ├── netflix_titles.csv                # Raw Netflix dataset

│   ├── netflix_labeled_family.csv        # Weakly labeled output from Notebook 1

│   └── netflix_augmented.csv             # Augmented dataset from Notebook 2

│

├── outputs/

│   ├── best_model.pkl                    # Saved model pipeline

│   ├── predictions_test.csv              # Model test predictions

│   └── figures/                          # ROC/PR curves, plots

│

├── notebooks/

│   ├── 01_Data_Labeling.ipynb            # Labeling & weak supervision


│   ├── 02_Data_Augmentation.ipynb        # NLP augmentation & balancing

│   └── 03_Model_Development.ipynb        # Slicing, modeling, evaluation

│

├── utils.py                              # Shared helper utilities

├── requirements.txt                      # Dependencies

└── README.md                             # Documentation (this file)



---

## 🧩 **Notebook 1 — Data Labeling**
**Objective:** Automatically label Netflix titles using heuristic rules and weak supervision.

**Key Steps:**
- Clean and preprocess raw data  
- Define label task `is_family_friendly`  
- Create **Labeling Functions (LFs)** based on:
  - Ratings (`G`, `PG`, `R`, `TV-Y`, etc.)
  - Genres (`listed_in`)
  - Keywords in `title` and `description`
- Aggregate LF votes (majority voting)
- Save labeled data → `data/netflix_labeled_family.csv`

**Outcome:**  
A weakly labeled dataset with family-friendly classifications ready for training.

---

## 🟨 **Notebook 2 — Data Augmentation**
**Objective:** Improve class balance and dataset diversity using NLP-based augmentation.

**Techniques Used:**
- **Synonym Replacement** — Replace words with WordNet synonyms  
- **Random Swap** — Swap positions of two random words  
- **Random Deletion** — Randomly remove non-critical words  

**Steps:**
1. Load weakly labeled data  
2. Identify minority class  
3. Generate augmented samples  
4. Merge and deduplicate  
5. Save to → `data/netflix_augmented.csv`

**Outcome:**  
A balanced, diverse dataset that strengthens model generalization.

---

## 🟦 **Notebook 3 — Data Slicing & Model Development**
**Objective:** Train robust models and evaluate performance across interpretable data slices.

**Pipeline Design:**
- **Text Features:** TF-IDF Vectorizer (bi-grams, 60k max features)  
- **Numeric Features:** Duration, release year, cast & genre counts  
- **Categorical Features:** Type, Rating, Duration Unit  
- Unified preprocessing via `ColumnTransformer`

**Models Compared:**
1. **Logistic Regression** — Linear, interpretable baseline  
2. **Random Forest** — Ensemble capturing nonlinear patterns  
3. **Calibrated Linear SVC** — Margin-based classifier with probability calibration  

**Evaluation Metrics:**
- 5-Fold **ROC-AUC** cross-validation  
- Test set **Accuracy**, **Precision**, **Recall**, **F1**, **ROC**, and **PR curves**  
- Slice-level metrics for:
  - Kids / Family genres  
  - Horror / Thriller content  
  - Long TV Shows (≥3 seasons)  
  - Short Movies (<90 minutes)  
  - Titles containing kids-related keywords  

**Outcome:**
- High-performing model (`best_model.pkl`)  
- ROC-AUC ≈ 0.99  
- Balanced performance across slices  
- Predictions file → `outputs/predictions_test.csv`

---

## 🧮 **Technologies Used**
| Category | Libraries / Tools |
|-----------|-------------------|
| **Language** | Python 3.10+ |
| **Data Handling** | pandas, numpy |
| **Text Processing** | scikit-learn (TF-IDF), nltk (WordNet) |
| **Modeling** | LogisticRegression, RandomForest, Calibrated LinearSVC |
| **Evaluation** | sklearn.metrics (ROC, PR, F1, Confusion Matrix) |
| **Visualization** | matplotlib, seaborn |
| **Serialization** | joblib |
| **Explainability (optional)** | shap |

---

## ⚙️ **Setup Instructions**

### 1️⃣ Clone or Download the Project
```bash
git clone https://github.com/<your-username>/netflix_lab.git
cd netflix_lab
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebooks in order:

```
01_Data_Labeling.ipynb → 02_Data_Augmentation.ipynb → 03_Model_Development.ipynb
```

---

## 📈 **Results Summary**

| Metric                          | Score       |
| :------------------------------ | :---------- |
| **ROC-AUC (5-Fold CV)**         | 0.99        |
| **Test ROC-AUC**                | 0.999       |
| **Precision / Recall (Family)** | 0.94 / 0.92 |
| **Slice Avg Accuracy**          | > 0.93      |

✅ Excellent model consistency and near-perfect ROC / PR curves demonstrate successful weak-label learning and generalization.

---

## 🧠 **Key Learnings**

* **Weak Supervision** allows high-quality label generation with minimal manual effort.
* **Lightweight Augmentation** helps balance small, biased datasets.
* **Slice-Based Evaluation** is essential for fairness and explainability.
* Combining **TF-IDF text features** with interpretable models yields reliable, explainable classification results.

---

## 📦 **Outputs**

After running all three notebooks, you’ll have:

| File                              | Description                         |
| --------------------------------- | ----------------------------------- |
| `data/netflix_labeled_family.csv` | Weakly labeled dataset              |
| `data/netflix_augmented.csv`      | Augmented dataset                   |
| `outputs/best_model.pkl`          | Final trained model pipeline        |
| `outputs/predictions_test.csv`    | Model predictions and probabilities |
| `outputs/figures/`                | Saved ROC/PR curves and slice plots |

---

## 🧰 **Utilities (utils.py)**

Common helper functions for all notebooks:

* `load_netflix_data()` – Load CSV safely
* `clean_colnames()` – Standardize column names
* `parse_duration()` – Convert duration text to numeric
* `top_n_counter()` – Count top genres or countries
* `add_genre_flags()` – One-hot encode top genres
* `save_fig()` – Save plots to `outputs/figures/`

---

## 📦 **Dependencies**

Refer to `requirements.txt`:

```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.1
scikit-learn>=1.4.0
joblib>=1.3.2
shap>=0.44.0
jupyter>=1.0.0
ipykernel>=6.25.0
scipy>=1.11.0
tqdm>=4.66.0
```
