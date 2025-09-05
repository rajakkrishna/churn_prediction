# 📊 Customer Churn Prediction

This project predicts **customer churn** using multiple supervised machine learning models:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- KNN  
- Support Vector Classifier  

The dataset is cleaned (duplicates removed, null values imputed) and preprocessed with **Label Encoding** and **Scaling**.  
The best trained model is saved with **joblib** and deployed using **Streamlit** for interactive predictions.  

### Features
- User-friendly Streamlit app with dropdowns for categorical data and inputs for numerical data.  
- Displays both **prediction** and **probability scores**.  
- Preprocessing (encoders & scaler) are saved and reused to avoid mismatches.  
