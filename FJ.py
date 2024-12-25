import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\pythonprojects\fakejob.csv")

# Display the first few rows to check for missing values
print(df.head())

# Handle missing values in 'description' column
df = df.dropna(subset=['description'])  # Drop rows with missing descriptions
# Alternatively, use this line if you want to fill missing values with a placeholder:
# df['description'] = df['description'].fillna('No description')

# Assume that 'description' is the text column and 'fraudulent' is the label (1 for fake, 0 for real)
X = df['description']  # Features (job descriptions)
y = df['fraudulent']  # Target (1 = Fake, 0 = Real)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the job descriptions into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing the performance using a Confusion Matrix
conf_matrix = np.array([[np.sum((y_test == 0) & (y_pred == 0)), np.sum((y_test == 0) & (y_pred == 1))],
                        [np.sum((y_test == 1) & (y_pred == 0)), np.sum((y_test == 1) & (y_pred == 1))]])

plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Real', 'Fake'])
plt.yticks([0, 1], ['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predicting a new job listing
new_job = ["Looking for a highly skilled software developer for an exciting opportunity at a growing tech company."]
new_job_tfidf = vectorizer.transform(new_job)
prediction = model.predict(new_job_tfidf)
print("\nPrediction for the new job listing:", "Fake" if prediction[0] == 1 else "Real")
