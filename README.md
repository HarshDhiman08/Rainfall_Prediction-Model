# Rainfall Prediction using Machine Learning

This project is an interactive machine learning application designed to predict the likelihood of rainfall based on various meteorological parameters.
It uses a combination of data preprocessing, classification models, and visualization tools to deliver accurate predictions.

The project includes a Jupyter Notebook UI built with ipywidgets, allowing users to adjust weather conditions dynamically and receive real-time rainfall predictions.

🧠 Key Features

✅ Data Cleaning & Preprocessing:

Handles missing values automatically.

Encodes categorical variables like yes/no.

Removes irrelevant columns for efficiency.

✅ Model Training:

Implements Logistic Regression, XGBoost, and Support Vector Machine (SVM).

Uses RandomOverSampler to handle class imbalance.

Normalizes features with StandardScaler.

✅ Performance Evaluation:

Computes ROC-AUC scores for model comparison.

Displays Confusion Matrix for best-performing model.

✅ Interactive UI with ipywidgets:

Dynamically generate sliders for each input feature.

Real-time rainfall prediction output with confidence percentage.

✅ Best Model Selection:

Automatically selects the most accurate model based on AUC performance.

💻 Technologies Used
Category	Tools/Libraries
Programming Language	Python 3.x
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn, XGBoost
Imbalance Handling	imblearn (RandomOverSampler)
UI Widgets	ipywidgets
Evaluation Metrics	ROC-AUC, Confusion Matrix
📊 Data Description

The dataset Rainfall.csv contains various meteorological attributes used to predict whether it will rain on a given day.

Column	Description
day	Date of observation
humidity	Atmospheric moisture percentage
pressure	Air pressure (hPa)
wind_speed	Average wind speed (km/h)
temperature	Average temperature (°C)
rainfall	Target variable (1 = Yes, 0 = No)
cloud_cover	Percentage of cloud cover
evaporation	Rate of evaporation
...	Other relevant meteorological variables
🧮 Model Workflow
Data Collection → Data Cleaning → Feature Selection 
→ Train-Test Split → Class Balancing → Feature Scaling
→ Model Training (LR, SVM, XGBoost) → Evaluation
→ Interactive Prediction Interface

🧩 Code Highlights
🔹 Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(kernel='rbf', probability=True)
}

🔹 Handling Imbalanced Data
ros = RandomOverSampler(random_state=22)
X_train, y_train = ros.fit_resample(X_train, y_train)

🔹 Confusion Matrix Visualization
ConfusionMatrixDisplay.from_estimator(best_model, X_val, y_val)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

🔹 Interactive Widget UI
predict_btn = widgets.Button(
    description="🔮 Predict Rainfall",
    button_style='success'
)

🧩 How to Run the Project
Step 1 — Clone Repository
git clone https://github.com/<harshdhiman08>/Rainfall_Prediction.git
cd Rainfall-Prediction-ML

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Launch Jupyter Notebook
jupyter notebook

Step 4 — Run the Notebook

Open Rainfall_Prediction.ipynb and execute all cells sequentially.
Use the interactive sliders to input weather values and predict rainfall.

🧾 Requirements.txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
ipywidgets
jupyter

📈 Sample Output

Console Output Example:

✅ Data cleaned successfully!
Logistic Regression ROC-AUC: 0.8725
XGBoost ROC-AUC: 0.9032
SVM ROC-AUC: 0.8911

🏆 Best Model: XGBoost


Widget Output Example:

🌦️ Yes, it will likely RAIN today! (Confidence: 82.56%)

🧠 Project Learnings

How to perform end-to-end data preprocessing for real-world datasets.

Applying multiple ML algorithms and comparing them using ROC-AUC.

Building an interactive ML interface using Jupyter’s ipywidgets.

Handling imbalanced datasets with RandomOverSampler.

Deploying a user-friendly predictive system for decision support.

🚀 Future Improvements

Integrate a real-time weather API (like OpenWeatherMap).

Deploy as a web app using Streamlit or Flask.

Add feature importance visualization for explainability.

Use deep learning (LSTM) for time-series rainfall prediction.

👨‍💻 Author

Harsh Dhiman
📍 MCA Student | iOS & Flutter Developer | Data Science Enthusiast
🔗 github.com,HarshDhiman08

🔗 linkdin.com/HarshDhiman08

🏁 Conclusion

This project successfully demonstrates the power of machine learning in meteorological prediction.
By combining data preprocessing, model training, and interactive visualization, the system enables users to forecast rainfall with high accuracy and confidence — a valuable step toward smarter environmental analysis and agriculture planning.
