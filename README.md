# Evans & B&M Merchant Classification

Classify merchant transaction descriptions into **Evans**, **B&M**, or **other** using regex rules and ML/NLP baselines (TF-IDF + linear models).

## Project layout

```
├── Evans_BM_Merchant_Classification.ipynb   # Main analysis and models
├── data/
│   ├── train_set.csv                        # Training set (desc, MCC, tag)
│   └── test_set.csv                         # Held-out test set
├── docs/
│   └── InterviewAssignment.docx             # Original assignment brief
├── requirements.txt                         # Python dependencies (see below)
└── README.md
```

## Dependencies

All packages used in the notebook are listed in [`requirements.txt`](requirements.txt):

- **pandas**, **numpy** — data loading and manipulation  
- **matplotlib** — plots  
- **scikit-learn** — TF-IDF, models, metrics, clustering  
- **joblib** — model utilities (imported in notebook)  
- **jupyter** — run the notebook locally  

Install everything in one step:

```bash
pip install -r requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `Evans_BM_Merchant_Classification.ipynb` and run all cells. Paths assume you run Jupyter from the **repository root**.

## Data

| Column | Description |
|--------|-------------|
| `desc` | Merchant description text |
| `MCC`  | Merchant category code / info |
| `tag`  | Label: `evans`, `b & m`, or `other` |

## Approach (summary)

1. Exploratory analysis (class balance, duplicates, MCC usage)
2. Regex classifier for Evans and B&M
3. TF-IDF + Logistic Regression / Linear SVM comparison
4. Optional clustering within the `other` class

Details and results are in the notebook.

## Author

Kashif Moin — [GitHub](https://github.com/KashifMoin1410)
