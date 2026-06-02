# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# PHASE 1: Data Acquisition
# ---------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv('telco_churn.csv')

# Display the first few rows and basic info
print(f"Dataset Shape: {df.shape}\n")
print("Data Types and Non-Null Counts:")
print(df.info())

# ---------------------------------------------------------
# PHASE 2: Data Cleaning & Preprocessing
# ---------------------------------------------------------

# 1. Drop irrelevant identifiers
# customerID doesn't hold predictive power; it's just a unique string.
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)
    print("\nDropped 'customerID' column.")

# 2. Fix Hidden Missing Values
# In this specific dataset, 'TotalCharges' is often loaded as an 'object' (string) 
# because missing values are represented as empty spaces (" "). 
# We must force it to numeric.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many NaNs were created by the coercion
missing_charges = df['TotalCharges'].isnull().sum()
print(f"\nMissing TotalCharges found after conversion: {missing_charges}")

# Since it's usually a very small number (around 11 rows), it's safe to drop them
df.dropna(subset=['TotalCharges'], inplace=True)

# 3. Target Variable Inspection
# Let's see our class imbalance (how many people churned vs. stayed)
print("\nTarget Variable Distribution (Churn):")
print(df['Churn'].value_counts(normalize=True) * 100)

# Optional: Plot the imbalance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='viridis')
plt.title('Customer Churn Distribution')
plt.show()

print("\nPhase 1 & 2 Complete! Data is loaded and clean.")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PHASE 3: Feature Engineering & Splitting
# ---------------------------------------------------------
print("\n--- Starting Phase 3: Feature Engineering ---")

# 1. Convert Target Variable 'Churn' to binary (1 = Churned, 0 = Stayed)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 2. Map Binary Categorical Features to 0 and 1
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 3. One-Hot Encoding for multi-class categorical variables
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# 4. Feature Scaling for Neural Networks
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Split into Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 6. Train/Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print("Phase 3 Complete! Data is mathematically ready for the models.")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# PHASE 4: Baseline Machine Learning Models
# ---------------------------------------------------------
print("\n--- Starting Phase 4: Baseline Modeling ---")

# 1. Logistic Regression (The Industry Standard Baseline)
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

print("\n[Baseline 1] Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, log_preds) * 100:.2f}%")
# The classification report shows precision, recall, and f1-score
print(classification_report(y_test, log_preds))

# 2. Random Forest Classifier (Handles non-linear patterns)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n[Baseline 2] Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")
print(classification_report(y_test, rf_preds))

print("Phase 4 Complete! Baselines established.")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------------------------------------------------
# PHASE 5: Deep Learning Pipeline (Neural Network)
# ---------------------------------------------------------
print("\n--- Starting Phase 5: Deep Learning ---")

# 1. Initialize the Artificial Neural Network
nn_model = Sequential()

# 2. Add the Input Layer and First Hidden Layer
# input_dim matches your 30 features. We use 'relu' activation for hidden layers.
nn_model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
# Dropout randomly disables 20% of neurons during training to prevent overfitting
nn_model.add(Dropout(0.2)) 

# 3. Add a Second Hidden Layer
nn_model.add(Dense(units=32, activation='relu'))
nn_model.add(Dropout(0.2))

# 4. Add the Output Layer
# units=1 and activation='sigmoid' because we are predicting a strict binary outcome (1 or 0)
nn_model.add(Dense(units=1, activation='sigmoid'))

# 5. Compile the Model
# 'adam' is the industry-standard optimizer. 'binary_crossentropy' is the math used to calculate loss.
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train the Neural Network
print("Training Neural Network (50 Epochs)... this might take a minute...")
# We use validation_split to monitor how well it's learning on unseen data during training
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# 7. Evaluate the Model on the Test Data
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)

print("\n[Phase 5] Deep Learning Neural Network Results:")
print(f"Accuracy: {nn_accuracy * 100:.2f}%")

print("Phase 5 Complete! Deep Learning model trained.")
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# PHASE 6: Advanced Optimization & SMOTE (Target: 90%+)
# ---------------------------------------------------------
print("\n--- Starting Phase 6: Hyperparameter Optimization & SMOTE ---")

# 1. Apply SMOTE to balance the training data perfectly (50/50 split)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Old Training Target Shape: \n{y_train.value_counts()}")
print(f"New SMOTE Training Target Shape: \n{y_train_smote.value_counts()}")

# 2. Build an Optimized Deep Learning Model
opt_model = Sequential()

# Wider layers to capture more complex synthetic patterns
opt_model.add(Dense(units=128, activation='relu', input_dim=X_train_smote.shape[1]))
opt_model.add(Dropout(0.3)) # Increased dropout to prevent overfitting on synthetic data

opt_model.add(Dense(units=64, activation='relu'))
opt_model.add(Dropout(0.3))

opt_model.add(Dense(units=32, activation='relu'))
opt_model.add(Dropout(0.2))

opt_model.add(Dense(units=1, activation='sigmoid'))

opt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Add Early Stopping (Stops training automatically when accuracy peaks)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 4. Train the Optimized Model on the SMOTE data
print("Training Optimized Neural Network...")
opt_history = opt_model.fit(
    X_train_smote, y_train_smote, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stop],
    verbose=0
)

# 5. Evaluate the Optimized Model
opt_loss, opt_accuracy = opt_model.evaluate(X_test, y_test, verbose=0)

print("\n[Phase 6] Optimized Deep Learning Results:")
print(f"Final Optimized Accuracy: {opt_accuracy * 100:.2f}%")
print("Phase 6 Complete! Pipeline Optimization finished.")
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PHASE 7: Business Insights (Feature Importance)
# ---------------------------------------------------------
print("\n--- Starting Phase 7: Business Insights ---")

# 1. Get feature importances from the Random Forest model
importances = rf_model.feature_importances_

# 2. Get the names of the features
feature_names = X_train.columns

# 3. Create a DataFrame to store the results
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# 4. Sort the DataFrame by importance in descending order
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# 5. Print the top 10 most important features
print("\nTop 10 Drivers of Customer Churn:")
print(feature_importances_df.head(10).to_string(index=False))

# 6. (Optional) Visualize the Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df.head(10), palette='viridis')
plt.title('Top 10 Drivers of Customer Churn')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print("\nPhase 7 Complete! Business insights extracted.")
# ---------------------------------------------------------
# PHASE 8: Making Predictions on New Data (Inference)
# ---------------------------------------------------------
print("\n--- Starting Phase 8: Predicting a New Customer ---")

# 1. Create a fake "New Customer" profile
# We will create a profile that looks historically risky (Month-to-month, high bill, no support)
new_customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,                 # Only been a customer for 2 months
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',         # No tech support
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',# High risk contract
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.50,     # Expensive bill
    'TotalCharges': 191.00
}

# Convert this single dictionary into a Pandas DataFrame
new_df = pd.DataFrame([new_customer])

# 2. Apply the EXACT same preprocessing as Phase 3
# Map binary columns to 1s and 0s
new_df['gender'] = new_df['gender'].map({'Male': 1, 'Female': 0})
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    new_df[col] = new_df[col].map({'Yes': 1, 'No': 0})

# One-Hot Encode categorical columns
new_df = pd.get_dummies(new_df)

# 3. The "Alignment Trick" (Crucial Pro-Step)
# Because this is only 1 row of data, get_dummies won't create all 30 columns 
# that the training set had. We force the new dataframe to perfectly match 
# the training features (X_train.columns), filling any missing ones with 0.
new_df = new_df.reindex(columns=X_train.columns, fill_value=0)

# 4. Feature Scaling (The Trap)
# DO NOT use fit_transform! We only use .transform() so it scales this new data
# based on the math it already learned from the historical training data.
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
new_df[num_cols] = scaler.transform(new_df[num_cols])

# 5. Make the Prediction!
# The neural network outputs a probability between 0.0 and 1.0
prediction_prob = opt_model.predict(new_df, verbose=0)[0][0]

print("\n[Prediction Results for New Customer]")
print(f"Probability of Churning: {prediction_prob * 100:.2f}%")

# 6. Translate the math into Business Logic
if prediction_prob > 0.50:
    print("Business Action: HIGH RISK! Automatically flag for a 20% retention discount email.")
else:
    print("Business Action: Safe. Customer is likely to stay. No action required.")

print("\nPipeline Complete!")