from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


#logistic regression:
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Ensure df is available (it should be from previous cells)
# If running this cell independently, you might need to recreate df:
# data = {
#     'cancer patient name' : ['raju','rani','ravi','ramu'],
#     'cancer' : ['yes','no','no','yes']
# }
# df = pd.DataFrame(data)

# 1. Encode the target variable 'cancer' into numerical values (0 or 1)
# 'no' will be 0, 'yes' will be 1
le = LabelEncoder()
df['cancer_encoded'] = le.fit_transform(df['cancer'])

print("DataFrame with encoded target:\n", df[['cancer', 'cancer_encoded']])

# 2. Define Features (X) and Target (y)
# For this very simple example, we'll use the DataFrame index as a placeholder feature.
# In a real-world scenario, X would be actual relevant numerical features.
X = df.index.values.reshape(-1, 1) # Reshape for scikit-learn
y = df['cancer_encoded']

print("\nFeatures (X) - DataFrame Index:\n", X)
print("Target (y) - Encoded Cancer Status:\n", y)

# 3. Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y) # Training the model

# 4. Display the model's coefficients and intercept
print("\n--- Logistic Regression Model Results ---")
print(f"Coefficient: {model.coef_[0][0]:.4f}") # How much the feature influences the log-odds of the outcome
print(f"Intercept: {model.intercept_[0]:.4f}") # The log-odds when the feature is zero

print("\nThis model essentially describes the relationship between the DataFrame index (our placeholder feature) and the probability of having cancer.")




le = LabelEncoder()
df['cancer_encoded'] = le.fit_transform(df['cancer'])

X = df.index.values.reshape(-1, 1)
y = df['cancer_encoded']

model = LogisticRegression()
model.fit(X, y)

df['prediction'] = model.predict(X)

print(df)



# confusin matrix:
from sklearn.metrics import confusion_matrix
y_pred = df['prediction']
cm = confusion_matrix(y, y_pred)
print(cm)


# accuracy and precision:

from sklearn.metrics import accuracy_score, precision_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
