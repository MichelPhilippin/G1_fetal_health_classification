import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# --- 1. Load Data ---
df = pd.read_csv('dataset/fetal_health.csv')

# --- 2. Preprocessing ---
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# Scale the data (Crucial for higher accuracy!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 3. Advanced Modeling (Gradient Boosting) ---
# We train two models to show comparison in the report
print("Training Models...")

# Model A: Random Forest (Baseline)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

# Model B: Gradient Boosting (The "Pro" Model)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
acc_gb = accuracy_score(y_test, gb_model.predict(X_test))

print(f"\nRandom Forest Accuracy: {acc_rf*100:.2f}%")
print(f"Gradient Boosting Accuracy: {acc_gb*100:.2f}%") 

print("\n--- Detailed Classification Report (Gradient Boosting) ---")
print(classification_report(y_test, gb_model.predict(X_test), 
                            target_names=['Normal', 'Suspect', 'Pathological']))
# --- 4. Advanced Visualization: ROC Curve ---
# This graph is essential for high-level bioinfo projects
y_train_bin = label_binarize(y_train, classes=[1, 2, 3])
y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
n_classes = 3

# Fit a One-vs-Rest classifier for ROC analysis
clf = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42))
y_score = clf.fit(X_train, y_train_bin).decision_function(X_test)

# Plot
plt.figure(figsize=(10, 7))
colors = ['blue', 'red', 'green']
class_names = ['Normal', 'Suspect', 'Pathological']

# Calculate ROC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Multi-Class Performance')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# --- 5. Feature Importance (Updated for GB) ---
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(gb_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top 10 Clinical Features (Gradient Boosting Model)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.show()