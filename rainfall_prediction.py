
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay
import ipywidgets as widgets
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Rainfall.csv')
df.rename(str.strip, axis='columns', inplace=True)

# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical columns
df.replace({'yes':1, 'no':0}, inplace=True)

# Drop unnecessary columns if present
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True, errors='ignore')

print("✅ Data cleaned successfully!")

X = df.drop(['day', 'rainfall'], axis=1)
y = df['rainfall']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Handle class imbalance
ros = RandomOverSampler(random_state=22)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = {
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(kernel='rbf', probability=True)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:,1]
    auc = metrics.roc_auc_score(y_val, val_preds)
    results[name] = auc
    print(f"{name} ROC-AUC: {auc:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")

ConfusionMatrixDisplay.from_estimator(best_model, X_val, y_val)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()


title = widgets.HTML("<h2 style='color:#2E8B57'>🌧️ Rainfall Prediction Interface</h2>")
inputs = {}
sliders = []

# Create sliders dynamically
for col in X.columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    mean_val = float(df[col].mean())
    slider = widgets.FloatSlider(
        value=mean_val,
        min=min_val,
        max=max_val,
        step=(max_val - min_val)/100,
        description=col,
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='90%')
    )
    inputs[col] = slider
    sliders.append(slider)

# Predict button
predict_btn = widgets.Button(
    description="🔮 Predict Rainfall",
    button_style='success',
    layout=widgets.Layout(width='50%', margin='10px 0 0 0')
)

# Output display area
output = widgets.Output()

def on_predict(b):
    with output:
        clear_output()
        input_data = np.array([[inputs[col].value for col in X.columns]])
        input_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_scaled)[0]
        prob = best_model.predict_proba(input_scaled)[0][1]
        
        if prediction == 1:
            print(f"🌦️ Yes, it will likely RAIN today! (Confidence: {prob*100:.2f}%)")
        else:
            print(f"☀️ No, it will likely NOT rain today. (Confidence: {(1-prob)*100:.2f}%)")

predict_btn.on_click(on_predict)

# Display UI
display(widgets.VBox([title] + sliders + [predict_btn, output]))
