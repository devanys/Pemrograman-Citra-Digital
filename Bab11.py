def bab11_ekstraksi_fitur():
    print("\n" + "="*80)
    print("BAB 11: EKSTRAKSI FITUR DAN REPRESENTASI CITRA")
    print("="*80)
    
    print("\n[11.1] Gray Level Co-occurrence Matrix (GLCM)")
    print("-" * 40)
    print("Persamaan GLCM:")
    print("P(i,j,d,θ) = #{(x,y), (x',y') | f(x,y)=i, f(x',y')=j, d,θ}")
    print("di mana d = jarak, θ = arah")
    print("\nFitur tekstur dari GLCM:")
    print("Contrast: Σ_{i,j} (i-j)² · P(i,j)")
    print("Energy:   Σ_{i,j} P(i,j)²")
    print("Homogeneity: Σ_{i,j} P(i,j) / (1 + |i-j|)")
    print("Correlation: Σ_{i,j} (i-μ)(j-μ)P(i,j) / σ²")
    
    citra_smooth = np.ones((128, 128), dtype=np.uint8) * 128
    
    citra_texture = np.zeros((128, 128), dtype=np.uint8)
    for i in range(0, 128, 8):
        for j in range(0, 128, 8):
            citra_texture[i:i+4, j:j+4] = 0
            citra_texture[i+4:i+8, j+4:j+8] = 255
    
    citra_random = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    
    def ekstrak_fitur_glcm(citra, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        citra_q = (citra / 32).astype(np.uint8)
        
        glcm = graycomatrix(citra_q, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        return {'contrast': contrast, 'energy': energy, 
                'homogeneity': homogeneity, 'correlation': correlation}
    
    fitur_smooth = ekstrak_fitur_glcm(citra_smooth)
    fitur_texture = ekstrak_fitur_glcm(citra_texture)
    fitur_random = ekstrak_fitur_glcm(citra_random)
    
    print("\nPerbandingan fitur GLCM:")
    print(f"{'Fitur':<15} {'Smooth':<12} {'Texture':<12} {'Random':<12}")
    print("-" * 51)
    for k in fitur_smooth.keys():
        print(f"{k:<15} {fitur_smooth[k]:<12.4f} {fitur_texture[k]:<12.4f} {fitur_random[k]:<12.4f}")
    
    print("\n[11.1] Local Binary Pattern (LBP)")
    print("-" * 40)
    print("Persamaan LBP:")
    print("LBP(x,y) = Σ_p s(g_p - g_c) · 2^p")
    print("di mana s(x) = 1 jika x ≥ 0, else 0")
    print("      g_c = intensitas pusat")
    print("      g_p = intensitas tetangga ke-p")
    
    def lbp_manual(citra, radius=1, n_points=8):
        rows, cols = citra.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = citra[i, j]
                lbp_value = 0
                
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(round(i + radius * np.sin(angle)))
                    y = int(round(j + radius * np.cos(angle)))
                    
                    if citra[x, y] >= center:
                        lbp_value += 2**p
                
                lbp[i, j] = lbp_value
        
        return lbp
    lbp_smooth = local_binary_pattern(citra_smooth, P=8, R=1, method='uniform')
    lbp_texture = local_binary_pattern(citra_texture, P=8, R=1, method='uniform')

    print("\n[11.3] Principal Component Analysis (PCA)")
    print("-" * 40)
    print("Persamaan PCA:")
    print("1. Standardisasi: z = (x - μ) / σ")
    print("2. Covariance matrix: C = (1/n) X^T X")
    print("3. Eigen decomposition: C v = λ v")
    print("4. Proyeksi: Y = X V_k (k komponen utama)")

    n_samples = 100
    X = np.random.randn(n_samples, 10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 11: Ekstraksi Fitur', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_smooth, cmap='gray')
    axes[0,0].set_title('Citra Smooth')
    
    axes[0,1].imshow(citra_texture, cmap='gray')
    axes[0,1].set_title('Citra Texture')
    
    axes[0,2].imshow(lbp_texture, cmap='gray')
    axes[0,2].set_title(f'LBP Texture\nΣ s(g_p-g_c)·2^p')

    axes[1,0].hist(lbp_smooth.flatten(), bins=26, alpha=0.7, label='Smooth', color='blue')
    axes[1,0].hist(lbp_texture.flatten(), bins=26, alpha=0.7, label='Texture', color='red')
    axes[1,0].legend()
    axes[1,0].set_title('Histogram LBP')

    axes[1,1].bar(range(1, 11), pca.explained_variance_ratio_, alpha=0.7)
    axes[1,1].plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='Cumulative')
    axes[1,1].set_xlabel('Principal Component')
    axes[1,1].set_ylabel('Explained Variance Ratio')
    axes[1,1].set_title('PCA: Explained Variance')
    axes[1,1].legend()

    scatter = axes[1,2].scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(n_samples), cmap='viridis')
    axes[1,2].set_xlabel('PC1')
    axes[1,2].set_ylabel('PC2')
    axes[1,2].set_title('PCA Projection (2D)')
    plt.colorbar(scatter, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('bab11_ekstraksi_fitur.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 11 disimpan")
