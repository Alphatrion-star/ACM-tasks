# Loan Approval Prediction – Colab + Streamlit

## 1. How to Use (Colab + Temporary Web App)

This project is meant to run in **Google Colab**, and the Streamlit web app is **temporary** (it only works while the Colab runtime is running).

To use the web interface:

1. Upload `Loan_prediction.ipynb` to Google Drive and open it in **Google Colab**.
2. Run all cells in order:
   - Data loading, preprocessing, model training (Logistic Regression + Random Forest).
   - Cells that save the final model pipeline(s) and create `app.py`.
   - Cells that start Streamlit and ngrok.
3. At the end, Colab prints a **public ngrok URL**.
   - Click this URL (it opens an ngrok page).
   - Then click **“Visit Site”**.
   - The Streamlit **loan prediction web app** opens in your browser.
4. The link works **only while Colab is active**. When the runtime stops, the URL dies; to use it again you must re-run the notebook and get a new link.

---

## 2. Project Overview

This project builds **two loan approval classifiers** on a real loan dataset:

- A **Logistic Regression** model.
- A **Random Forest** model tuned with GridSearchCV.

Both use a common preprocessing pipeline. The notebook compares their performance, then integrates a chosen model into a **Streamlit web app** so users can interactively predict whether a loan will be **Approved (1)** or **Not Approved (0)** from applicant and loan details.

---

## 3. Dataset and Features

You work with a loan CSV where each row is one loan application.

- **Raw input features:**
  - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
  - `ApplicantIncome`, `CoapplicantIncome`
  - `LoanAmount`, `Loan_Amount_Term`
  - `Credit_History`
  - `Property_Area`

- **Target:**
  - `Loan_Status` – `Y` (approved) / `N` (rejected), encoded to 1 / 0.

- **Engineered features:**
  - `Total_Income` = ApplicantIncome + CoapplicantIncome
  - `Loan_to_Income_ratio` = LoanAmount / Total_Income
  - `EMI_to_Income_ratio` ≈ (monthly EMI) / Total_Income

`Loan_ID` is dropped as it is only an identifier.

---

## 4. Preprocessing and Models

### 4.1 Preprocessing Pipeline

A scikit-learn **Pipeline** with a **ColumnTransformer** handles preprocessing for both models:

- **Numeric features** (incomes, loan amounts, ratios, binary flags, etc.):
  - Median imputation (`SimpleImputer`).
  - MinMax scaling.

- **Categorical features** (`Dependents`, `Property_Area`):
  - Most-frequent imputation.
  - One-hot encoding (`OneHotEncoder(handle_unknown='ignore')`).

This pipeline feeds into both the Logistic Regression and Random Forest classifiers.

### 4.2 Logistic Regression

- Trained on the preprocessed features to estimate approval probability.
- Evaluated using:
  - Accuracy.
  - Precision, recall, F1-score.
  - Confusion matrix.

### 4.3 Random Forest + Hyperparameter Tuning

- A **Random Forest classifier** is trained using the same preprocessing pipeline.
- Hyperparameters (number of trees, max depth, min samples split/leaf, criterion) are tuned with **GridSearchCV**, using ROC-AUC as the scoring metric.
- A small results table is printed showing accuracy for various tree counts; the tuned Random Forest reaches accuracy in the mid‑80% range on the test set.

The notebook compares Logistic Regression vs Random Forest and then selects one (commonly the tuned Random Forest) as the final model for deployment.

---

## 5. Streamlit Web Application (via Colab)

### 5.1 Saving the Final Pipeline

After training and selecting the best model, the notebook:

- Saves the full pipeline (preprocessing + classifier) to disk using `pickle` (e.g., `model_log.pkl` or `model_rf.pkl`).

### 5.2 `app.py` (High-Level Behavior)

The notebook writes an `app.py` script that:

- Loads the chosen pipeline.
- Recomputes engineered features from user inputs:
  - `Total_Income`, `Loan_to_Income_ratio`, `EMI_to_Income_ratio`.
- Provides a Streamlit UI with inputs for:
  - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
  - `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
  - `Credit_History`, `Property_Area`
- On **Predict**:
  - Builds a one-row DataFrame matching the training feature order.
  - Runs `predict` and `predict_proba` on the pipeline.
  - Shows:
    - **Loan Approved / Loan Not Approved**.
    - Estimated approval probability.

The app can use either the Logistic Regression or Random Forest pipeline depending on which is saved and loaded.

### 5.3 Business Rules & Validation

The app adds simple rule checks to keep predictions realistic, for example:

- Auto-reject if `Total_Income` is 0.
- Auto-reject if `Loan_to_Income_ratio` is extremely high.
- Optionally auto-approve very “safe” profiles.

These guard against clearly unrealistic inputs that the model never saw during training.

### 5.4 Running via ngrok in Colab

- Streamlit runs on a local port inside Colab.
- ngrok exposes that port through a temporary public URL.
- You click the URL, then **“Visit Site”**, to open the app.
- When the Colab runtime ends, the app and URL stop working and must be restarted.

---

## 6. Typical Prediction Errors and Limitations

Even with Logistic Regression and a tuned Random Forest, some errors are expected:

**Borderline approvals/rejections**  
   Applicants near the decision (like medium income/loan size, borderline credit history) can be misclassified:
   - True approvals predicted as rejections (FN).
   - True rejections predicted as approvals (FP).



