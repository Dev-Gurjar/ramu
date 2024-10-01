# Popular Person Identification with Streamlit

This project identifies popular persons from uploaded images using SVM and Naive Bayes classifiers, with PCA and LDA for feature reduction.

## Features
- **Upload** an image (JPG/PNG) for classification.
- Uses **PCA** and **LDA** for feature extraction.
- Predicts identity using **SVM** and **Naive Bayes** models.
- Displays the predicted person's name.

## Setup

1. Install required libraries:
    ```bash
    pip install streamlit joblib numpy pillow pandas scikit-learn
    ```
2. Ensure the following models are in the directory:
   - `svm_model.pkl`, `NB_model.pkl`, `pca_model.pkl`, `lda_model.pkl`
3. Add `target_data.csv` with a column `person_name` for class labels.
4. Run the app:
    ```bash
    streamlit run app.py
    ```

## File Structure
