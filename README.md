## Heart Disease Risk Predictor


This project predicts the likelihood of heart disease using clinical features and machine learning. By utilizing an XGBoost classifier and 13 standard clinical heart health indicators, the system estimates whether a patient falls into a low or high-risk category. This tool is designed to provide early awareness and educational insights—it serves as a technical demonstration of predictive modeling in healthcare and is not a substitute for professional medical diagnosis.


---


## Method Details

-**Initialization:** Set up the Python environment and load essential libraries (Pandas, Scikit-Learn, XGBoost).

-**Data Preprocessing:** Handle categorical encoding (e.g., chest pain types) and apply feature scaling to prevent bias toward large numerical values.

-**Model Training:** Train a baseline model before moving into recursive feature elimination and hyperparameter optimization.

-**Evaluation:** Assess the model's ability to minimize "False Negatives" (crucial in medical contexts) using a Confusion Matrix.

-**Persistence:** Export the model as a .pkl or .joblib file to ensure the predictor can be used without retraining.

-**Prediction:** Feed raw patient data into the pipeline to receive a structured risk assessment.

<img width="628" height="349" alt="image" src="https://github.com/user-attachments/assets/465ee7be-4d10-48e7-bec2-3a5820e743fe" />
