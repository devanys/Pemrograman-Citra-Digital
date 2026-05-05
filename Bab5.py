def bab5_filtering():
    print("\n" + "="*80)
    print("BAB 5: OPERASI SPASIAL DAN FILTERING")
    print("="*80)

    print("\n[5.1] Konsep Konvolusi 2D")
    print("-" * 40)
    print("Persamaan konvolusi:")
    print("g(x,y) = Σ_m Σ_n h(m,n) · f(x-m, y-n)")
    print("di mana h = kernel/filter, f = citra input")
    
    def konvolusi_2d(citra, kernel):
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        padded = np.pad(citra, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        output = np.zeros_like(citra, dtype=np.float64)
        
        for i in range(citra.shape[0]):
            for j in range(citra.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    print("\n[5.2] Smoothing Filters")
    print("-" * 40)

    kernel_mean = np.ones((3, 3)) / 9
    print("Mean Filter 3×3:")
    print(kernel_mean)

    def gaussian_kernel(size, sigma):
        x, y = np.mgrid[-size//2:size//2+1, -size//2:size//2+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    kernel_gaussian = gaussian_kernel(3, 1.0)
    print("\nGaussian Filter 3×3 (σ=1):")
    print(kernel_gaussian)
    print("\n[5.3] Sharpening Filter - Laplacian")
    print("-" * 40)
    print("Persamaan Laplacian:")
    print("∇²f = ∂²f/∂x² + ∂²f/∂y²")
    print("      ≈ f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)")
    
    kernel_laplacian = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])
    print("\nKernel Laplacian:")
    print(kernel_laplacian)
    
    def sharpening_laplacian(citra):
        laplacian = konvolusi_2d(citra.astype(float), kernel_laplacian)
        sharpened = citra.astype(float) - laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)
      
    print("\n[5.4] Median Filter")
    print("-" * 40)
    print("g(x,y) = median{f(x-i, y-j) | (i,j) ∈ W}")
    print("di mana W = window sekitar (x,y)")
    
    def median_filter_manual(citra, size=3):
        pad = size // 2
        padded = np.pad(citra, pad, mode='edge')
        output = np.zeros_like(citra)
        
        for i in range(citra.shape[0]):
            for j in range(citra.shape[1]):
                window = padded[i:i+size, j:j+size]
                output[i, j] = np.median(window)
        
        return output.astype(np.uint8)
    
    citra_bersih = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(citra_bersih, (50, 50), (200, 200), 200, -1)
    cv2.circle(citra_bersih, (125, 125), 50, 100, -1)
    
    noisy_gaussian = citra_bersih.astype(float) + np.random.normal(0, 30, citra_bersih.shape)
    noisy_gaussian = np.clip(noisy_gaussian, 0, 255).astype(np.uint8)
    
    noisy_sp = citra_bersih.copy()
    mask = np.random.random(citra_bersih.shape) < 0.05
    noisy_sp[mask] = np.random.choice([0, 255], size=mask.sum())
    
    smoothed_mean = konvolusi_2d(noisy_gaussian.astype(float), kernel_mean)
    smoothed_gaussian = konvolusi_2d(noisy_gaussian.astype(float), kernel_gaussian)
    sharpened = sharpening_laplacian(citra_bersih)
    median_filtered = median_filter_manual(noisy_sp)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 5: Operasi Spasial dan Filtering', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(noisy_gaussian, cmap='gray')
    axes[0,0].set_title('Citra + Gaussian Noise')
    
    axes[0,1].imshow(smoothed_mean, cmap='gray')
    axes[0,1].set_title('Mean Filter\nΣ/9')
    
    axes[0,2].imshow(smoothed_gaussian, cmap='gray')
    axes[0,2].set_title('Gaussian Filter\nG(x,y) = exp(-(x²+y²)/2σ²)')
    
    axes[1,0].imshow(noisy_sp, cmap='gray')
    axes[1,0].set_title('Citra + Salt & Pepper')
    
    axes[1,1].imshow(median_filtered, cmap='gray')
    axes[1,1].set_title('Median Filter\nmedian{W}')
    
    axes[1,2].imshow(sharpened, cmap='gray')
    axes[1,2].set_title('Sharpening (Laplacian)\ng = f - ∇²f')
    
    plt.tight_layout()
    plt.savefig('bab5_filtering.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 5 disimpan")
