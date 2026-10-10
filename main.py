import numpy as np
import matplotlib.pyplot as plt

from dataPrep import Encodare_train, Encodare_real
from svd import SVD_trunchiat

FISIER_TRAIN = "kddcup.data_10_percent_corrected"
FISIER_TEST  = "corrected"
K            = 20

print("[1] Incarc datele de antrenament...")
X_train, medie, deviatie, coloane = Encodare_train(FISIER_TRAIN)
print(f"    Matrice train: {X_train.shape}")

print(f"[2] Calculez SVD trunchiat cu k={K}...")
U, S, Vt = SVD_trunchiat(X_train, K)
print(f"    Valori singulare: {np.round(S, 2)}")

print("[3] Calculez pragul theta...")

def eroare_reconstructie(X, Vt_mat):
    proiectat = X @ Vt_mat.T @ Vt_mat
    erori = np.sum((X - proiectat) ** 2, axis=1)
    return erori

erori_train = eroare_reconstructie(X_train, Vt)

p999 = np.percentile(erori_train, 99.9)
erori_train_clean = erori_train[erori_train < p999]
theta = np.percentile(erori_train_clean, 95)
print(f"    Theta (prag): {theta:.4f}")

print("[4] Incarc datele de test...")
X_test, etichete = Encodare_real(FISIER_TEST, medie, deviatie, coloane)
print(f"    Matrice test: {X_test.shape}")

erori_test = eroare_reconstructie(X_test, Vt)
predictii = np.where(erori_test > theta, "atac", "normal")
etichete_reale = np.where(etichete == "normal.", "normal", "atac")

corecte  = np.sum(predictii == etichete_reale)
total    = len(etichete_reale)
acuratete = corecte / total * 100

n_atacuri_reale     = np.sum(etichete_reale == "atac")
n_atacuri_detectate = np.sum((predictii == "atac") & (etichete_reale == "atac"))
n_fals_pozitive     = np.sum((predictii == "atac") & (etichete_reale == "normal"))

print("\n─── REZULTATE ─────────────────────────────────────────────")
print(f"    Acuratete totala   : {acuratete:.2f}%")
print(f"    Atacuri detectate  : {n_atacuri_detectate} / {n_atacuri_reale}")
print(f"    Fals pozitive      : {n_fals_pozitive}")


print("\n[5] Generez graficul...")

normale_idx = etichete_reale == "normal"
atacuri_idx = etichete_reale == "atac"

MAX_EROARE = theta * 5
erori_viz = np.clip(erori_test, 0, MAX_EROARE)

plt.figure(figsize=(12, 5))
plt.scatter(
    np.where(normale_idx)[0], erori_viz[normale_idx],
    s=2, color="steelblue", alpha=0.5, label="Normal"
)
plt.scatter(
    np.where(atacuri_idx)[0], erori_viz[atacuri_idx],
    s=2, color="tomato", alpha=0.5, label="Atac"
)
plt.axhline(y=theta, color="black", linewidth=1.5,
            linestyle="--", label=f"Prag θ = {theta:.2f}")

plt.xlabel("Index pachet")
plt.ylabel("Eroare de reconstructie")
plt.title("Detectie anomalii prin SVD trunchiat")
plt.legend()
plt.tight_layout()
plt.savefig("rezultat.png", dpi=150)
plt.show()
print("    Grafic salvat ca rezultat.png")

