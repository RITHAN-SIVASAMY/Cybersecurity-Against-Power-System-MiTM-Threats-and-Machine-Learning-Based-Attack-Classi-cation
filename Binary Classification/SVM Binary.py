import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
data_path = r"C:\Users\ritha\OneDrive\Desktop\2ND SEMESTER\EOCS MITM\4.csv"
data = pd.read_csv(data_path)

# Debug: Inspect the dataset
print("Dataset Info:\n", data.info())
print("First Few Rows:\n", data.head())

# Drop unnecessary columns if they exist
data = data.drop(columns=[col for col in ['Unnamed: 0', 'Time'] if col in data.columns])

# Handle missing values: fill numerical columns with mean
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Drop categorical columns that are entirely NaN
data = data.dropna(axis=1, how='all')

# Recalculate categorical columns after dropping empty ones
categorical_cols = data.select_dtypes(include=['object']).columns

# Encode categorical variables
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing values with mode
    data[col] = LabelEncoder().fit_transform(data[col])  # Encode categories

# Ensure the target column exists
if 'snort_alert' not in data.columns:
    raise ValueError("The target column 'snort_alert' is missing in the dataset.")

# Identify features and target variable
X = data.drop(columns=['snort_alert'])
y = data['snort_alert']

# Debug: Check correlation with target to avoid data leakage
print("Feature Correlation with Target:\n", data.corr()['snort_alert'].sort_values(ascending=False))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Debug: Ensure train-test split is valid
print("Unique labels in training set:", set(y_train))
print("Unique labels in testing set:", set(y_test))

# Scale features (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier with a radial basis function (RBF) kernel
clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print and visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - SVM")
plt.show()

# Save the trained model
joblib.dump(clf, 'svm_intrusion_model.pkl')
