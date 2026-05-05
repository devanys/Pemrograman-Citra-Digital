def bab1_konsep_dasar():
    print("\n" + "="*80)
    print("BAB 1: KONSEP DASAR DAN SISTEM PENGOLAHAN CITRA")
    print("="*80)

    print("\n[1.2.1] Definisi Citra Kontinu")
    print("-" * 40)
    print("Citra kontinu direpresentasikan sebagai fungsi f(x,y):")
    print("  f: R² → R")
    print("  di mana x,y = koordinat spasial")
    print("  f(x,y) = intensitas/amplitudo pada titik (x,y)")
  
    print("\n[1.2.2] Citra Digital - Fungsi Diskret")
    print("-" * 40)
    print("Citra digital didefinisikan sebagai:")
    print("  f(x,y) : {0,1,...,M-1} × {0,1,...,N-1} → {0,1,...,L-1}")
    print("  M = jumlah baris, N = jumlah kolom")
    print("  L = 2^k = jumlah level intensitas (k = bit depth)")
    
    # Implementasi citra digital sederhana
    M, N, k = 4, 4, 8  # 4x4 piksel, 8-bit
    L = 2**k  # L = 256 level intensitas
    citra_contoh = np.array([
        [50,  100, 150, 200],
        [75,  125, 175, 225],
        [25,   75, 125, 175],
        [0,    50, 100, 150]
    ], dtype=np.uint8)
    
    print(f"\nContoh matriks citra digital {M}x{N} (8-bit):")
    print(citra_contoh)
    print(f"\nRentang intensitas: [0, {L-1}]")
    print(f"Total konfigurasi berbeda: L^(M×N) = {L**(M*N):.2e}")
    
    print("\n[Pers. 1.1] Structural Similarity Index (SSIM)")
    print("-" * 40)
    print("SSIM(x,y) = (2μ_x·μ_y + C₁)(2σ_xy + C₂) / ((μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂))")
    print("di mana:")
    print("  μ_x, μ_y = nilai rata-rata")
    print("  σ_x², σ_y² = variansi")
    print("  σ_xy = kovarians")
    print("  C₁ = (K₁·L)², C₂ = (K₂·L)² (konstanta stabilisasi)")
    
    def hitung_ssim(img1, img2, L=256, K1=0.01, K2=0.03):
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        
        mu_x = np.mean(img1)
        mu_y = np.mean(img2)
        
        sigma_x_sq = np.var(img1)
        sigma_y_sq = np.var(img2)
        sigma_xy = np.cov(img1.flatten(), img2.flatten())[0, 1]
        
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim_value = numerator / denominator
        return ssim_value
    
    img_original = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    img_noisy = img_original.astype(np.float64) + np.random.normal(0, 10, img_original.shape)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    ssim_val = hitung_ssim(img_original, img_noisy)
    print(f"\nSSIM antara citra asli dan noisy: {ssim_val:.6f}")
    
    print("\n[Pers. 1.2] Entropi Shannon")
    print("-" * 40)
    print("H = -Σ p(r_k) · log₂(p(r_k))  untuk k = 0, 1, ..., L-1")
    print("di mana:")
    print("  p(r_k) = probabilitas kemunculan intensitas r_k")
    print("  L = jumlah total level intensitas")
    
    def hitung_entropi(citra):
        histogram, _ = np.histogram(citra.flatten(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum() 
        
        p_rk = histogram[histogram > 0]
        
        entropi = -np.sum(p_rk * np.log2(p_rk))
        return entropi
    
    citra_kontras_rendah = np.ones((64, 64), dtype=np.uint8) * 128  
    citra_kontras_tinggi = np.random.randint(0, 256, (64, 64), dtype=np.uint8)  
    
    entropi_rendah = hitung_entropi(citra_kontras_rendah)
    entropi_tinggi = hitung_entropi(citra_kontras_tinggi)
    
    print(f"\nEntropi citra kontras rendah (uniform): {entropi_rendah:.4f} bit")
    print(f"Entropi citra kontras tinggi (random):  {entropi_tinggi:.4f} bit")
    print(f"Entropi maksimum (8-bit):               8.0000 bit")
    
    print("\n[1.2.2] Representasi Citra Warna RGB")
    print("-" * 40)
    print("f(x,y) = [f_R(x,y), f_G(x,y), f_B(x,y)]ᵀ")
    print("Memori untuk citra RGB: Memori = M × N × 3 × k bits")
    
    M, N, k = 1000, 1000, 8 
    memori_bytes = M * N * 3 * (k // 8)
    print(f"\nContoh: Citra {M}×{N} RGB {k}-bit")
    print(f"Memori tanpa kompresi: {memori_bytes:,} bytes = {memori_bytes/1024/1024:.2f} MB")
    print("\n[1.3.1] Hounsfield Unit (CT Scan)")
    print("-" * 40)
    print("HU = (μ - μ_water) / μ_water × 1000")
    print("di mana:")
    print("  μ = koefisien atenuasi jaringan")
    print("  μ_water = koefisien atenuasi air")
    
    def hounsfield_unit(mu_jaringan, mu_air=1.0):
        return (mu_jaringan - mu_air) / mu_air * 1000
    
    jaringan = {
        'Udara': 0.0,
        'Lemak': 0.9,
        'Air': 1.0,
        'Jaringan Lunak': 1.04,
        'Tulang Kortikal': 1.9
    }
    
    print("\nNilai Hounsfield Unit untuk berbagai jaringan:")
    for nama, mu in jaringan.items():
        hu = hounsfield_unit(mu)
        print(f"  {nama:20s}: μ = {mu:.2f}, HU = {hu:+.0f}")

    print("\n[1.3.2] Teorema Sampling Nyquist-Shannon")
    print("-" * 40)
    print("f_s ≥ 2 × f_max")
    print("di mana:")
    print("  f_s = frekuensi sampling")
    print("  f_max = frekuensi maksimum sinyal")
    
    def cek_nyquist(f_max, f_s):
        f_nyquist = 2 * f_max
        terpenuhi = f_s >= f_nyquist
        return terpenuhi, f_nyquist
    
    f_max = 100  # Hz
    f_s = 250    # Hz
    
    terpenuhi, f_nyquist = cek_nyquist(f_max, f_s)
    print(f"\nFrekuensi maksimum sinyal: {f_max} Hz")
    print(f"Frekuensi Nyquist minimum: {f_nyquist} Hz")
    print(f"Frekuensi sampling: {f_s} Hz")
    print(f"Kondisi Nyquist terpenuhi: {terpenuhi}")

    print("\n[1.3.2] Perhitungan Resolusi Spasial")
    print("-" * 40)
    print("Total Piksel = (Lebar_cm × PPI) × (Tinggi_cm × PPI)")
    print("Memori (bit) = M × N × k")
    
    def hitung_resolusi(lebar_cm, tinggi_cm, ppi, bit_depth=8):
        M = int(lebar_cm / 2.54 * ppi)  # Konversi cm ke inch
        N = int(tinggi_cm / 2.54 * ppi)
        total_piksel = M * N
        memori_bits = total_piksel * bit_depth
        memori_bytes = memori_bits / 8
        return M, N, total_piksel, memori_bytes
    
    lebar, tinggi, ppi = 20, 15, 300  # Foto 20x15 cm, 300 PPI
    M, N, total, memori = hitung_resolusi(lebar, tinggi, ppi)
    print(f"\nFoto {lebar}×{tinggi} cm pada {ppi} PPI:")
    print(f"  Resolusi: {M} × {N} piksel")
    print(f"  Total piksel: {total:,}")
    print(f"  Memori (8-bit grayscale): {memori/1024/1024:.2f} MB")
    print(f"  Memori (24-bit RGB): {memori*3/1024/1024:.2f} MB")
    
    print("\n[1.4.1] Segmentasi sebagai Klasifikasi Piksel")
    print("-" * 40)
    print("P(y|I_x) = Probabilitas posterior piksel x termasuk kelas y")
    print("  I_x = nilai intensitas piksel x")
    print("  y ∈ {1, 2, ..., K} = label kelas")
    
    # Visualisasi konsep dasar
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 1: Konsep Dasar Citra Digital', fontsize=14, fontweight='bold')
    
    # 1. Matriks citra digital
    axes[0,0].imshow(citra_contoh, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title('Matriks Citra Digital 4×4')
    for i in range(4):
        for j in range(4):
            axes[0,0].text(j, i, str(citra_contoh[i,j]), ha='center', va='center', 
                          color='white' if citra_contoh[i,j] < 128 else 'black', fontsize=8)
    
    # 2. Histogram dan Entropi
    axes[0,1].hist(citra_kontras_tinggi.flatten(), bins=256, color='blue', alpha=0.7)
    axes[0,1].set_title(f'Histogram (Entropi={entropi_tinggi:.2f} bit)')
    axes[0,1].set_xlabel('Intensitas')
    axes[0,1].set_ylabel('Frekuensi')
    
    # 3. Representasi RGB
    citra_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    citra_rgb[:, :33, 0] = 255  # Merah
    citra_rgb[:, 33:66, 1] = 255  # Hijau
    citra_rgb[:, 66:, 2] = 255  # Biru
    axes[0,2].imshow(citra_rgb)
    axes[0,2].set_title('Representasi RGB')
    
    # 4. Koneksi Piksel
    koneksi_4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    koneksi_8 = np.ones((3,3))
    axes[1,0].imshow(koneksi_4, cmap='gray')
    axes[1,0].set_title('Koneksi-4')
    for i in range(3):
        for j in range(3):
            axes[1,0].text(j, i, str(int(koneksi_4[i,j])), ha='center', va='center')
    
    axes[1,1].imshow(koneksi_8, cmap='gray')
    axes[1,1].set_title('Koneksi-8')
    for i in range(3):
        for j in range(3):
            axes[1,1].text(j, i, str(int(koneksi_8[i,j])), ha='center', va='center')
    
    # 5. SSIM visualization
    axes[1,2].imshow(np.abs(img_original.astype(float) - img_noisy.astype(float)), cmap='hot')
    axes[1,2].set_title(f'Diferensi Citra (SSIM={ssim_val:.4f})')
    
    plt.tight_layout()
    plt.savefig('bab1_konsep_dasar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 1 disimpan: bab1_konsep_dasar.png")
