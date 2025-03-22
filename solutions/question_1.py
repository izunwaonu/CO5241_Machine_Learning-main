import numpy as np

# Function to calculate entropy
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Training dataset (CreditScore and RiskLevel)
data = [
    (720, "Low"),
    (650, "High"),
    (750, "Low"),
    (600, "High"),
    (780, "Low"),
    (630, "High"),
    (710, "Low"),
    (640, "High"),
]

# Split data based on CreditScore <= 650
left_split = [risk for credit, risk in data if credit <= 650]
right_split = [risk for credit, risk in data if credit > 650]

# Calculate entropy before split
labels = [risk for _, risk in data]
H_before = entropy(labels)

# Calculate entropy after split
H_left = entropy(left_split)
H_right = entropy(right_split)

# Calculate weighted entropy
total = len(data)
H_after = (len(left_split) / total) * H_left + (len(right_split) / total) * H_right

# Compute Information Gain
IG = H_before - H_after

# Print results
print(f"Entropy before split: {H_before:.4f}")
print(f"Entropy after split: {H_after:.4f}")
print(f"Information Gain: {IG:.4f}")

# Decision: Is CreditScore=650 a good split?
if IG > 0.5:
    print(" CreditScore=650 is a good split!")
else:
    print(" Consider a different feature for splitting.")
