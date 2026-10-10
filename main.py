import numpy as np
import matplotlib.pyplot as plt

from dataPrep import encode_train, encode_real
from svd import truncated_SVD

TRAIN_FILE = "kddcup.data_10_percent_corrected"
TEST_FILE  = "corrected"
K          = 20

print("[1] Loading training data...")
X_train, mean, std, columns = encode_train(TRAIN_FILE)
print(f"    Train matrix: {X_train.shape}")

print(f"[2] Computing truncated SVD with k={K}...")
U, S, Vt = truncated_SVD(X_train, K)
print(f"    Singular values: {np.round(S, 2)}")

print("[3] Computing threshold theta...")

def reconstruction_error(X, Vt_mat):
    projected = X @ Vt_mat.T @ Vt_mat
    errors = np.sum((X - projected) ** 2, axis=1)
    return errors

train_errors = reconstruction_error(X_train, Vt)

p999 = np.percentile(train_errors, 99.9)
train_errors_clean = train_errors[train_errors < p999]
theta = np.percentile(train_errors_clean, 95)
print(f"    Theta (threshold): {theta:.4f}")

print("[4] Loading test data...")
X_test, labels = encode_real(TEST_FILE, mean, std, columns)
print(f"    Test matrix: {X_test.shape}")

test_errors = reconstruction_error(X_test, Vt)
predictions = np.where(test_errors > theta, "attack", "normal")
true_labels = np.where(labels == "normal.", "normal", "attack")

correct  = np.sum(predictions == true_labels)
total    = len(true_labels)
accuracy = correct / total * 100

n_real_attacks     = np.sum(true_labels == "attack")
n_detected_attacks = np.sum((predictions == "attack") & (true_labels == "attack"))
n_false_positives  = np.sum((predictions == "attack") & (true_labels == "normal"))

print("\n─── RESULTS ───────────────────────────────────────────────")
print(f"    Overall accuracy   : {accuracy:.2f}%")
print(f"    Attacks detected   : {n_detected_attacks} / {n_real_attacks}")
print(f"    False positives    : {n_false_positives}")


print("\n[5] Generating plot...")

normal_idx = true_labels == "normal"
attack_idx = true_labels == "attack"

MAX_ERROR = theta * 5
errors_viz = np.clip(test_errors, 0, MAX_ERROR)

plt.figure(figsize=(12, 5))
plt.scatter(
    np.where(normal_idx)[0], errors_viz[normal_idx],
    s=2, color="steelblue", alpha=0.5, label="Normal"
)
plt.scatter(
    np.where(attack_idx)[0], errors_viz[attack_idx],
    s=2, color="tomato", alpha=0.5, label="Attack"
)
plt.axhline(y=theta, color="black", linewidth=1.5,
            linestyle="--", label=f"Threshold θ = {theta:.2f}")

plt.xlabel("Packet index")
plt.ylabel("Reconstruction error")
plt.title("Anomaly detection using truncated SVD")
plt.legend()
plt.tight_layout()
plt.savefig("result.png", dpi=150)
plt.show()
print("    Plot saved as result.png")