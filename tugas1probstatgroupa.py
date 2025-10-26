import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Data yang Anda berikan
data_dict = {
    'Kabupaten/Kota': [
        'Kepulauan Seribu',
        'Kota Jakarta Selatan',
        'Kota Jakarta Timur',
        'Kota Jakarta Pusat',
        'Kota Jakarta Barat',
        'Kota Jakarta Utara',
    ],
    'Jemaah Haji': [3, 1803, 2855, 575, 1468, 1181]
}

# Buat DataFrame dari dictionary
df = pd.DataFrame(data_dict)
print("📊 Menggunakan data yang di-declare secara manual...")

# Pastikan kolom sesuai
df.columns = df.columns.str.strip()
data = df[["Kabupaten/Kota", "Jemaah Haji"]].copy()

# (Bagian ini tidak perlu diubah, tetap berfungsi)
data["Jemaah Haji"] = pd.to_numeric(data["Jemaah Haji"], errors="coerce")
data = data.dropna(subset=["Jemaah Haji"])

# Hitung statistik deskriptif
mean_val = data["Jemaah Haji"].mean()
median_val = data["Jemaah Haji"].median()
variance_val = data["Jemaah Haji"].var()
std_val = data["Jemaah Haji"].std()

print("\n📊 Statistik Jemaah Haji di DKI Jakarta:")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val}")
print(f"Variance: {variance_val:.2f}")
print(f"Standard Deviation: {std_val:.2f}")

# Buat histogram (FREKUENSI) + kurva KDE (diskalakan)
x = data["Jemaah Haji"]
num_bins = 6 # Tentukan jumlah bin

# 1. Tentukan jumlah data (N)
N = len(x)

# 2. Tentukan lebar bin (bin_width)
# (max - min) / jumlah bin
# Tambahkan nilai kecil (epsilon) untuk menghindari pembagian dengan nol jika datanya sama semua
bin_width = (x.max() - x.min()) / num_bins
if bin_width == 0:
    bin_width = 1 # Atur default jika semua data sama

# 3. Hitung KDE (ini masih density)
kde = gaussian_kde(x)
x_grid = np.linspace(x.min(), x.max(), 200)
kde_values_density = kde(x_grid) # Ini adalah nilai density (total area = 1)

# 4. Skalakan KDE agar sesuai dengan frekuensi
# Skala = N * bin_width
kde_values_scaled = kde_values_density * N * bin_width

plt.figure(figsize=(9,6))

# 5. Plot histogram dengan FREKUENSI (counts)
# Hapus 'density=True'
plt.hist(
    x, bins=num_bins, # Gunakan num_bins yg sama
    alpha=0.4, color="skyblue",
    edgecolor="black", label="Histogram (Frekuensi)"
)

# 6. Plot KDE yang sudah DISKALAKAN
plt.plot(x_grid, kde_values_scaled, color="blue", linewidth=2, label="Estimasi Kurva (diskalakan)")

# Garis mean, median
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.0f}')
plt.text(mean_val - 50, plt.ylim()[1]*0.85, "Mean →", color="#E74C3C", fontsize=10, ha="right", verticalalignment='center')
plt.text(median_val + 50, plt.ylim()[1]*0.65, "← Median", color="#1F618D", fontsize=10, ha="left", verticalalignment='center')

# Hias grafik
plt.title("Histogram Jumlah Jemaah Haji per Kabupaten/Kota\nProvinsi DKI Jakarta Tahun 2024", fontsize=13, fontweight="bold")
plt.xlabel("Jumlah Jemaah Haji", fontsize=11)
plt.ylabel("Frekuensi (Jumlah Kabupaten/Kota)", fontsize=11)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
