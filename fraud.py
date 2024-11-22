import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

data = pd.read_csv(r'C:\Users\ADMIN\Desktop\credit card fraud detection\creditcard.csv') 

print(data.isnull().sum())

scaler = StandardScaler()
data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
      'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
      'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
      'Amount']] = scaler.fit_transform(data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                                              'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 
                                              'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
                                              'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])


print(data['Class'].value_counts())

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

log_reg = LogisticRegression(solver='liblinear', max_iter=500, class_weight='balanced', random_state=42)
log_reg.fit(X_train_smote, y_train_smote)

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_smote, y_train_smote)

y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("Logistic Regression - Classification Report:\n", classification_report(y_test, y_pred_log_reg))
print("Logistic Regression - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Logistic Regression - ROC AUC Score:", roc_auc_score(y_test, y_pred_log_reg))

print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest - ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))




