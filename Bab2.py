def bab2_akuisisi():
    print("\n" + "="*80)
    print("BAB 2: AKUISISI DAN REPRESENTASI CITRA")
    print("="*80)

    print("\n[2.2] Sampling dan Kuantisasi")
    print("-" * 40)
    
    x = np.linspace(0, 4*np.pi, 256)
    y = np.linspace(0, 4*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    citra_kontinu = (np.sin(X) * np.cos(Y) + 1) * 127.5  # Rentang 0-255
    
    def sampling_citra(citra, faktor):
        return citra[::faktor, ::faktor]
    
    def kuantisasi_citra(citra, level):
        L = 256
        step = L / level
        terkuantisasi = np.floor(citra / step) * step + step/2
        return np.clip(terkuantisasi, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('BAB 2: Sampling dan Kuantisasi', fontsize=14, fontweight='bold')
    
    faktor_sampling = [1, 2, 4, 8]
    for i, f in enumerate(faktor_sampling):
        sampled = sampling_citra(citra_kontinu, f)
        axes[0, i].imshow(sampled, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Sampling {256//f}×{256//f}\n(faktor {f})')
    
    level_kuantisasi = [256, 16, 4, 2]
    for i, level in enumerate(level_kuantisasi):
        terkuantisasi = kuantisasi_citra(citra_kontinu, level)
        axes[1, i].imshow(terkuantisasi, cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Kuantisasi {level} level\n({int(np.log2(level))} bit)')
    
    plt.tight_layout()
    plt.savefig('bab2_sampling_kuantisasi.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n[2.5] Model Noise")
    print("-" * 40)

    def tambah_noise_gaussian(citra, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, citra.shape)
        return np.clip(citra + noise, 0, 255).astype(np.uint8)
    
    def tambah_noise_impuls(citra, prob=0.05):
        noisy = citra.copy()
        salt = np.random.random(citra.shape) < prob/2
        noisy[salt] = 255
        pepper = np.random.random(citra.shape) < prob/2
        noisy[pepper] = 0
        return noisy
    
    def tambah_noise_speckle(citra, var=0.1):
        noise = np.random.normal(0, np.sqrt(var), citra.shape)
        return np.clip(citra + citra * noise, 0, 255).astype(np.uint8)

    def hitung_psnr(img1, img2):
        mse = np.mean((img1.astype(float) - img2.astype(float))**2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 10 * np.log10(max_pixel**2 / mse)
        return psnr
    
    citra_test = plt.imread('bab1_konsep_dasar.png')[:,:,0] if os.path.exists('bab1_konsep_dasar.png') else np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 2: Model Noise pada Citra', fontsize=14, fontweight='bold')
    
    noisy_gaussian = tambah_noise_gaussian(citra_test)
    noisy_impuls = tambah_noise_impuls(citra_test)
    noisy_speckle = tambah_noise_speckle(citra_test)
    
    axes[0,0].imshow(citra_test, cmap='gray')
    axes[0,0].set_title('Citra Asli')
    
    axes[0,1].imshow(noisy_gaussian, cmap='gray')
    axes[0,1].set_title(f'Gaussian Noise\nPSNR={hitung_psnr(citra_test, noisy_gaussian):.2f} dB')
    
    axes[0,2].imshow(noisy_impuls, cmap='gray')
    axes[0,2].set_title(f'Salt & Pepper Noise\nPSNR={hitung_psnr(citra_test, noisy_impuls):.2f} dB')
    
    axes[1,0].imshow(noisy_speckle, cmap='gray')
    axes[1,1].set_title(f'Speckle Noise\nPSNR={hitung_psnr(citra_test, noisy_speckle):.2f} dB')
    
    axes[1,1].hist(citra_test.flatten(), bins=50, alpha=0.5, label='Asli', color='blue')
    axes[1,1].hist(noisy_gaussian.flatten(), bins=50, alpha=0.5, label='Gaussian', color='red')
    axes[1,1].legend()
    axes[1,1].set_title('Perbandingan Histogram')
    
    axes[1,2].axis('off')
    axes[1,2].text(0.5, 0.5, 'Persamaan PSNR:\nPSNR = 10·log₁₀(MAX²/MSE)', 
                   ha='center', va='center', fontsize=12, transform=axes[1,2].transAxes)
    
    plt.tight_layout()
    plt.savefig('bab2_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 2 disimpan")
