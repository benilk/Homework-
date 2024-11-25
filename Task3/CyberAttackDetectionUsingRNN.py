import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
#Load DataSet
dataset_path = "kdd99.csv"
data = pd.read_csv(dataset_path)
print(data.head())

#PreProcess
label_encoder = LabelEncoder()

data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
data['service'] = label_encoder.fit_transform(data['service'])
data['flag'] = label_encoder.fit_transform(data['flag'])
data['label'] = label_encoder.fit_transform(data['label'])

X = data.drop(columns=['label'])
y = data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Shape of training data: {X_train.shape}")
print(f"Shape of test data: {X_test.shape}")

#RSNN
model = Sequential()

#Add LSTM
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#Compile
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

#Learn
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))

#Model
y_pred = model.predict(X_test_rnn)
y_pred = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.6, label="Actual")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred.flatten(), cmap="viridis", alpha=0.3, label="Predicted")

plt.title("Actual vs Predicted (Scatter Plot)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
