import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib

# Simulate training data: Create random 250x350 pixel images (flattened to 1D arrays)
# Let's create 1000 random "images" of 250x350 pixels (75,000 features)
X = np.random.rand(1000, 250 * 350)  # 1000 images with 250x350 pixels flattened into 1D
y = np.random.randint(0, 10, 1000)  # Random labels from 0 to 9 (digits)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Use PCA for dimensionality reduction (this can speed up training)
pca = PCA(n_components=40)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train an SVM classifier
model = SVC(gamma=0.001)
model.fit(X_train, y_train)

# Predict and check the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using joblib
joblib.dump(model, 'digit_recognition_model.pkl')
joblib.dump(pca, 'pca_transformer.pkl')  # Save PCA transformer to use later
