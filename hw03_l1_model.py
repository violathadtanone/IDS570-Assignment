from pathlib import Path
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")

with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

# Update penalty
clf = LogisticRegression(
    l1_ratio=1,
    solver="liblinear",
    max_iter=2000
)

clf.fit(X_train, y_train)

#test set predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

### ROC AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

# Model sparsity diagnostic
non_zero = (clf.coef_[0] != 0).sum()
print("Non-zero coefficients:", non_zero)

# Inspect what the model learned
feature_names = vectorizer.get_feature_names_out()
coefficients = clf.coef_[0]

# Sort coefficients from smallest to largest
sorted_indices = coefficients.argsort()

# Top negative-weight words
negative_indices = [i for i in sorted_indices if coefficients[i] < 0]
print("\nTop 15 negative-weight words (predictive of NEG = 0):")
for i in negative_indices[:15]:
    print(feature_names[i], coefficients[i])

# Top 15 positive-weight words
print("\nTop 15 positive-weight words (predictive of CORE = 1):")
for i in sorted_indices[-15:][::-1]:
    print(feature_names[i], coefficients[i])
    
# Inspect misclassified examples
false_positives = []
false_negatives = []

for text, true_label, pred_label in zip(X_test_texts, y_test, y_pred):
    if true_label == 0 and pred_label == 1:
        false_positives.append(text)

    elif true_label == 1 and pred_label == 0:
        false_negatives.append(text)


print("\n--- 5 FALSE POSITIVES ---")
for i, text in enumerate(false_positives[:5], 1):
    print(f"\nFalse Positive {i}:")
    print(text)


print("\n--- 5 FALSE NEGATIVES ---")
for i, text in enumerate(false_negatives[:5], 1):
    print(f"\nFalse Negative {i}:")
    print(text)