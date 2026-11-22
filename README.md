# Bank Term Deposit Subscription Prediction (UCI Machine Learning Repository)

## 1. Problem Statement

Direct telemarketing remains a primary channel through which commercial banking institutions promote term deposit subscriptions. However, unguided phone outreach incurs substantial operational costs, reduces agent productivity, and degrades customer goodwill through unwanted solicitations. The objective of this machine learning project is to construct, evaluate, and compare five distinct classification models to predict whether a prospective client will subscribe to a bank term deposit (target variable $y \in \{0, 1\}$, where $1 = \text{'yes'}$ and $0 = \text{'no'}$).

By accurately identifying high-propensity leads, financial institutions can optimize marketing allocation, maximize subscription conversion rates, and minimize unnecessary outbound communication overhead.

## 2. Dataset Choice & Comprehensive Description

### 2.1 Dataset Selection Justification

The **Bank Marketing Dataset** (sourced from the UCI Machine Learning Repository) was selected for this classification task based on the following criteria:

* **Benchmark Compliance:** Contains **16 input features** and **4,521 instances** (in `bank.csv`, or 45,211 in `bank-full.csv`), satisfying and exceeding the assignment requirements of at least 12 features and 1,000 instances.
* **Mixed Feature Modalities:** Incorporates continuous numeric metrics (`balance`, `duration`, `age`), discrete integer counts (`campaign`, `previous`), nominal categories (`job`, `marital`, `contact`), and ordinal attributes (`education`, `month`). This structural variety allows for a robust demonstration of end-to-end preprocessing pipelines (median/mode imputation, Z-score standardization, and One-Hot Encoding).
* **Realistic Missing Values:** Includes implicit missing values recorded as the categorical string `"unknown"` across attributes such as `job`, `education`, `contact`, and `poutcome`, requiring pipeline-integrated imputation.
* **Severe Class Imbalance:** Features an imbalanced target distribution ($\approx 88.48\%$ negative vs. $\approx 11.52\%$ positive), making it ideal for evaluating distribution-robust metrics such as **ROC-AUC**, **F1-Score**, and the **Matthews Correlation Coefficient (MCC)** alongside raw Accuracy.

### 2.2 Target Variable Identification

* **Target Column Name:** `y` (mapped to `deposit` in secondary mirrors).
* **Data Type:** Categorical / Binary String (`'yes'`, `'no'`).
* **Binary Encoding Scheme:**
  * $1$ ($\text{'yes'}$): Client subscribed to a term deposit (Positive Class).
  * $0$ ($\text{'no'}$): Client did not subscribe to a term deposit (Negative Class).

### 2.3 Feature Schema & Measurement Scales

| Feature Name | Data Type | Feature Class | Measurement Scale | Description |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Numeric | Continuous | Ratio | Age of the client in years |
| `job` | Categorical | Nominal | 12 Categories | Occupation (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`) |
| `marital` | Categorical | Nominal | 3 Categories | Marital status (`divorced`, `married`, `single`) |
| `education` | Categorical | Ordinal | 4 Levels | Highest education level (`primary`, `secondary`, `tertiary`, `unknown`) |
| `default` | Categorical | Binary | Nominal | Credit in default? (`yes`, `no`) |
| `balance` | Numeric | Continuous | Interval | Average yearly balance in Euros ($€$) |
| `housing` | Categorical | Binary | Nominal | Has active housing loan? (`yes`, `no`) |
| `loan` | Categorical | Binary | Nominal | Has active personal loan? (`yes`, `no`) |
| `contact` | Categorical | Nominal | 3 Categories | Communication contact type (`cellular`, `telephone`, `unknown`) |
| `day` | Numeric | Discrete | Ratio | Last contact day of the month ($1 \le \text{day} \le 31$) |
| `month` | Categorical | Ordinal | 12 Levels | Last contact month of the year (`jan` through `dec`) |
| `duration` | Numeric | Continuous | Ratio | Last contact duration in seconds |
| `campaign` | Numeric | Discrete | Ratio | Number of contacts performed during current campaign |
| `pdays` | Numeric | Discrete | Ratio | Days passed after client was last contacted ($-1$ indicates no prior contact) |
| `previous` | Numeric | Discrete | Ratio | Number of contacts performed prior to current campaign |
| `poutcome` | Categorical | Nominal | 4 Categories | Outcome of previous marketing campaign (`failure`, `other`, `success`, `unknown`) |

### 2.4 Class Distribution Examination

Analysis of the ground-truth target vector reveals pronounced class imbalance:

* **Total Dataset Instances ($N$):** $4,521$
* **Class 0 ($\text{'no'}$ - Non-subscribers):** $4,000$ instances ($\approx 88.48\%$)
* **Class 1 ($\text{'yes'}$ - Subscribers):** $521$ instances ($\approx 11.52\%$)
* **Imbalance Ratio:** $\approx 7.68 : 1$

#### Analytical Implications:

1. **Accuracy Paradox:** A naive dummy baseline predicting $0$ for every sample achieves an Accuracy of $88.48\%$ despite identifying zero actual deposit subscribers. Accuracy alone is insufficient for evaluating model quality.
2. **Metric Selection:** Model optimization prioritizes **Recall** (capturing potential subscribers), **Precision** (minimizing wasted contacts), **ROC-AUC** (discrimination ability across classification thresholds), and **Matthews Correlation Coefficient (MCC)** (measuring correlation across all confusion matrix quadrants resistant to class imbalance).
3. **Stratified Sampling:** An $80:20$ train-test split is executed with label stratification (`stratify=y`, `random_state=42`) to maintain the $88.48\% : 11.52\%$ ratio across both training ($N_{\text{train}} = 3,616$) and testing ($N_{\text{test}} = 905$) partitions.

## 3. GitHub Repository Information & Project Structure

* **GitHub Repository URL:** `https://github.com/<your-username>/bank-term-deposit-ml-classification`

```text
bank-term-deposit-ml-classification/
│
├── requirements.txt             # Required Python package dependencies
├── README.md                    # Project documentation & assignment report
├── test_data.csv                # Exported 20% test dataset partition (with true target column)
├── model_evaluation_metrics.png # Generated performance metric visualization plot
└── model/                       # Notebook & serialized trained model binaries
    ├── assignment_notebook.ipynb # Jupyter Notebook containing full source code
    ├── preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib
```

## 4. Execution Instructions

### Prerequisites

Ensure Python $3.9+$ is installed on your local environment or the BITS Virtual Lab.

### Step 1: Clone Repository & Install Dependencies

```bash
git clone https://github.com/<your-username>/bank-term-deposit-ml-classification.git
cd bank-term-deposit-ml-classification
pip install -r requirements.txt
```

### Step 2: Launch and Run Jupyter Notebook

1. Start the Jupyter Notebook environment:
   ```bash
   jupyter notebook
   ```
2. Open `model/assignment_notebook.ipynb` (or from your workspace root).
3. Execute all cells sequentially from top to bottom (**Kernel -> Restart & Run All**).
4. The notebook automatically:
   - Loads and cleans `bank.csv`.
   - Performs missing value imputation and stratified train-test splitting.
   - Exports `test_data.csv`.
   - Fits preprocessor pipelines **only on training data**.
   - Trains all 5 classifiers and serializes `.joblib` model binaries into `model/`.
   - Displays the evaluation summary table and outputs `model_evaluation_metrics.png`.

## 5. Machine Learning Models & Evaluation Metrics Comparison

Five classification algorithms were trained on the preprocessed feature matrix ($X_{\text{train}}$) and evaluated on the held-out test set ($X_{\text{test}}$). Preprocessing pipelines were fitted **exclusively on the training split** to prevent data leakage.

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8906 | 0.8715 | 0.5682 | 0.2381 | 0.3356 | 0.3235 |
| **Decision Tree** | 0.8950 | 0.8407 | 0.5849 | 0.2952 | 0.3924 | 0.3732 |
| **kNN** | 0.8873 | 0.7712 | 0.5312 | 0.1619 | 0.2482 | 0.2497 |
| **Naive Bayes** | 0.8265 | 0.7938 | 0.3333 | 0.5238 | 0.4074 | 0.3204 |
| **Random Forest (Ensemble)** | **0.9039** | **0.8982** | **0.6739** | **0.2952** | **0.4106** | **0.4071** |

## 6. Metrics Visualization (Graphical Representation)

The graph below represents the metric comparison across all 5 implemented models:

```text
Score
1.0 |                                                
0.9 |    LR   DT   RF    LR   DT   RF               RF
0.8 |    ██   ██   ██    ██   ██   ██   NB   NB     ██   LR   DT   RF
0.7 |    ██   ██   ██    ██   ██   ██   ██   ██     ██   ██   ██   ██
0.6 |    ██   ██   ██    ██   ██   ██   ██   ██     ██   ██   ██   ██
0.5 |    ██   ██   ██    ██   ██   ██   ██   ██     ██   ██   ██   ██
0.0 +------------------------------------------------------------------
      Accuracy          AUC          Precision    Recall      MCC
```

*(Note: The notebook saves a high-resolution grouped bar plot as `model_evaluation_metrics.png` for inclusion in your PDF submission).*

## 7. Model Performance Observations & Diagnosis

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers a solid parametric baseline with strong Accuracy ($0.8906$) and ROC-AUC ($0.8715$). However, because it fits a linear decision boundary, it struggles with non-linear feature interactions (e.g., non-linear relationships between `duration` and `pdays`), leading to a low Recall score of $0.2381$. |
| **Decision Tree** | Successfully captures non-linear split boundaries (such as call duration thresholds), yielding higher Recall ($0.2952$), F1 ($0.3924$), and MCC ($0.3732$) than Logistic Regression. However, a single decision tree remains susceptible to structural variance and slight overfitting on local feature subsets. |
| **kNN** | Yields the lowest performance metrics among all evaluated models (AUC: $0.7712$, Recall: $0.1619$, MCC: $0.2497$). Following One-Hot Encoding of categorical variables, the high-dimensional feature space causes distance metrics to suffer from the curse of dimensionality, diluting Euclidean distance calculations. |
| **Naive Bayes** | Achieves the highest Recall ($0.5238$) among all individual base models by identifying over half of true subscriber leads. However, its strong class-conditional feature independence assumption is violated by correlated features (e.g., `pdays` and `previous`), resulting in lower Precision ($0.3333$) and a higher false-positive rate. |
| **Random Forest (Ensemble)** | Achieves superior performance across every primary evaluation metric (Accuracy: $0.9039$, AUC: $0.8982$, Precision: $0.6739$, F1: $0.4106$, MCC: $0.4071$). By aggregating decorrelated decision trees, the ensemble reduces variance, captures complex feature interactions, and handles class imbalance better than standalone models. |

### Overall Winner for the Dataset

**Random Forest (Ensemble)** is the overall winner.

In imbalanced classification problems, raw Accuracy can be misleading. Random Forest achieved the highest **AUC ($0.8982$)**, **Precision ($0.6739$)**, **F1-Score ($0.4106$)**, and **MCC ($0.4071$)**. It offers the best balance between maximizing conversion captures and avoiding false-positive outreach costs.
