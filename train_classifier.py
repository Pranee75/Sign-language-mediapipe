import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data_list = data_dict['data']
label_list = data_dict['labels']

# Debug: Check a few entry lengths
for i, features in enumerate(data_list[:5]):
    print(f"Sample {i} length: {len(features)}")

# Auto-detect expected length from first entry (if not sure)
expected_length = len(data_list[0]) if len(data_list) > 0 else 0

# Filter valid entries
filtered_data = []
filtered_labels = []

for features, label in zip(data_list, label_list):
    if isinstance(features, (list, np.ndarray)) and len(features) == expected_length:
        filtered_data.append(features)
        filtered_labels.append(label)

print(f"Total valid samples after filtering: {len(filtered_data)}")
print("Unique class labels in filtered dataset:", set(filtered_labels))
print("Number of classes:", len(set(filtered_labels)))

if len(filtered_data) == 0:
    raise ValueError("No valid samples found. Check if your landmark data is being extracted correctly.")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(filtered_labels)
print("Classes learned by LabelEncoder:", label_encoder.classes_)

# Convert to NumPy arrays
data = np.array(filtered_data)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


