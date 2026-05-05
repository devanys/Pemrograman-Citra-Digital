def bab4_image_enhancement():
    print("\n" + "="*80)
    print("BAB 4: PENINGKATAN KUALITAS CITRA")
    print("="*80)
    
    np.random.seed(42)
    citra_kontras_rendah = np.random.normal(100, 30, (256, 256)).astype(np.uint8)
    print("\n[4.2] Transformasi Intensitas")
    print("-" * 40)
    print("Transformasi linear: s = T(r) = a·r + b")
    print("Transformasi log: s = c·log(1 + r)")
    print("Transformasi power (gamma): s = c·r^γ")
    
    def transformasi_linear(citra, a=1.0, b=0):
        return np.clip(a * citra + b, 0, 255).astype(np.uint8)
    
    def transformasi_log(citra, c=1.0):
        s = c * np.log1p(citra.astype(np.float64))
        s = (s / s.max() * 255).astype(np.uint8)
        return s
    
    def gamma_correction(citra, gamma=1.0, c=1.0):
        s = c * np.power(citra.astype(np.float64) / 255.0, gamma) * 255.0
        return np.clip(s, 0, 255).astype(np.uint8)
    
    print("\n[4.3] Histogram Equalization")
    print("-" * 40)
    print("Persamaan CDF (Cumulative Distribution Function):")
    print("  CDF(r_k) = Σ p(r_j) untuk j = 0, 1, ..., k")
    print("Transformasi: s_k = round((L-1) · CDF(r_k))")
    
    def histogram_equalization_manual(citra):
        L = 256
        histogram, _ = np.histogram(citra.flatten(), bins=L, range=(0, L))
        pdf = histogram / histogram.sum()
        cdf = np.cumsum(pdf)
        
        s_k = np.round((L - 1) * cdf).astype(np.uint8)
        
        citra_equalized = s_k[citra]
        return citra_equalized, histogram, cdf
    
    print("\n[4.4] Gamma Correction")
    print("-" * 40)
    print("s = c · (r/255)^γ")
    print("γ < 1: brightening (memperjelas gelap)")
    print("γ > 1: darkening (memperjelas terang)")

    citra_log = transformasi_log(citra_kontras_rendah, c=50)
    citra_equalized, hist_asli, cdf = histogram_equalization_manual(citra_kontras_rendah)
    citra_gamma_05 = gamma_correction(citra_kontras_rendah, gamma=0.5)
    citra_gamma_22 = gamma_correction(citra_kontras_rendah, gamma=2.2)
    citra_linear = transformasi_linear(citra_kontras_rendah, a=1.5, b=20)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('BAB 4: Peningkatan Kualitas Citra', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_kontras_rendah, cmap='gray')
    axes[0,0].set_title('Citra Asli\n(Kontras Rendah)')
    
    axes[0,1].imshow(citra_log, cmap='gray')
    axes[0,1].set_title('Transformasi Log\ns = c·log(1+r)')
    
    axes[0,2].imshow(citra_equalized, cmap='gray')
    axes[0,2].set_title('Histogram Equalization\ns = (L-1)·CDF(r)')
    
    axes[1,0].imshow(citra_gamma_05, cmap='gray')
    axes[1,0].set_title('Gamma Correction γ=0.5\n(Brightening)')
    
    axes[1,1].imshow(citra_gamma_22, cmap='gray')
    axes[1,1].set_title('Gamma Correction γ=2.2\n(Darkening)')
    
    axes[1,2].imshow(citra_linear, cmap='gray')
    axes[1,2].set_title('Transformasi Linear\ns = 1.5r + 20')
    
    axes[2,0].hist(citra_kontras_rendah.flatten(), bins=50, color='blue')
    axes[2,0].set_title('Histogram Asli')
    
    axes[2,1].hist(citra_equalized.flatten(), bins=50, color='green')
    axes[2,1].set_title('Histogram Setelah HE')

    axes[2,2].plot(np.arange(256), cdf, 'r-', linewidth=2)
    axes[2,2].set_xlabel('Intensitas')
    axes[2,2].set_ylabel('CDF')
    axes[2,2].set_title('Cumulative Distribution Function')
    axes[2,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('bab4_enhancement.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 4 disimpan")

