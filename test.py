import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---- Load two audio files ----
audio1, sr1 = librosa.load("low.mp3",duration=3)
audio2, sr2 = librosa.load("high.mp3")

# ---- Extract ONE simple feature: spectral centroid ----
# (It roughly tells how "bright/high" a sound is)

feature1 = np.mean(librosa.feature.spectral_centroid(y=audio1, sr=sr1))
feature2 = np.mean(librosa.feature.spectral_centroid(y=audio2, sr=sr2))

# ---- Create dataset ----
X = np.array([[feature1], [feature2]])
y = np.array([0, 1])   # 0 = low pitch, 1 = high pitch

# ---- Train simple ML model ----
model = LogisticRegression()
model.fit(X, y)

# ---- Test on a new file ----
test_audio, sr = librosa.load("test.mp3")
test_feature = np.mean(librosa.feature.spectral_centroid(y=test_audio, sr=sr))

prediction = model.predict([[test_feature]])

print("Prediction:", "High Pitch" if prediction[0] == 1 else "Low Pitch")
