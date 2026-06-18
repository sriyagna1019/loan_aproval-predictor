import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample loan data
data = {
    'Income': [25000, 40000, 50000, 60000, 70000, 80000],
    'Loan_Amount': [100000, 150000, 200000, 250000, 300000, 350000],
    'Approved': [0, 0, 0, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and Output
X = df[['Income', 'Loan_Amount']]
y = df['Approved']

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new applicant
new_applicant = [[65000, 250000]]
prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")