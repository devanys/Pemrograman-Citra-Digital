def bab8_segmentasi():

    print("\n" + "="*80)
    print("BAB 8: SEGMENTASI CITRA")
    print("="*80)

    print("\n[8.2] Thresholding")
    print("-" * 40)
    print("Global: g(x,y) = 1 jika f(x,y) > T, else 0")
    print("\nOtsu's Method:")
    print("Memaksimalkan variance antar kelas:")
    print("σ²_B(T) = ω₁(T)·ω₂(T)·[μ₁(T) - μ₂(T)]²")
    
    def otsu_threshold(citra):
        histogram, _ = np.histogram(citra.flatten(), bins=256, range=(0, 256))
        pdf = histogram / histogram.sum()
        
        best_threshold = 0
        max_variance = 0
        
        for T in range(256):
            w0 = np.sum(pdf[:T])
            w1 = np.sum(pdf[T:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.sum(np.arange(T) * pdf[:T]) / w0
            mu1 = np.sum(np.arange(T, 256) * pdf[T:]) / w1
            
            variance = w0 * w1 * (mu0 - mu1)**2
            
            if variance > max_variance:
                max_variance = variance
                best_threshold = T
        
        return best_threshold
    print("\n[8.3] Region Growing")
    print("-" * 40)
    print("Region bertumbuh jika:")
    print("|f(x,y) - μ_region| < T")
    
    def region_growing(citra, seed, threshold=10):
      
        rows, cols = citra.shape
        segmented = np.zeros_like(citra, dtype=np.uint8)
        region_pixels = [seed]
        segmented[seed] = 255
        region_mean = float(citra[seed])
        
        while region_pixels:
            current = region_pixels.pop(0)
            x, y = current
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if segmented[nx, ny] == 0:
                        if abs(int(citra[nx, ny]) - region_mean) < threshold:
                            segmented[nx, ny] = 255
                            region_pixels.append((nx, ny))
                            # Update mean
                            region_mean = (region_mean * np.sum(segmented == 255) + citra[nx, ny]) / (np.sum(segmented == 255) + 1)
        
        return segmented
    
    print("\n[8.4] K-Means Clustering")
    print("-" * 40)
    print("Minimisasi:")
    print("J = Σ_{k=1}^{K} Σ_{x∈C_k} ||x - μ_k||²")
    print("Update: μ_k = (1/|C_k|) Σ_{x∈C_k} x")
    
    def kmeans_segmentasi(citra, k=3, max_iter=100):
        pixels = citra.reshape(-1, 1).astype(float)
        centroids = np.random.choice(pixels.flatten(), k, replace=False).reshape(-1, 1)
        for _ in range(max_iter):
            distances = np.abs(pixels - centroids.T)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([pixels[labels == i].mean() for i in range(k)]).reshape(-1, 1)
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels.reshape(citra.shape)
    citra_segmentasi = np.zeros((256, 256), dtype=np.uint8)
    citra_segmentasi[:128, :128] = 50
    citra_segmentasi[:128, 128:] = 150
    citra_segmentasi[128:, :128] = 100
    citra_segmentasi[128:, 128:] = 200
    
    noisy = citra_segmentasi.astype(float) + np.random.normal(0, 15, citra_segmentasi.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    T_otsu = otsu_threshold(noisy)
    binary_otsu = (noisy > T_otsu).astype(np.uint8) * 255
    
    region = region_growing(noisy, seed=(50, 50), threshold=25)
    
    kmeans_result = kmeans_segmentasi(noisy, k=4)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 8: Segmentasi Citra', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(noisy, cmap='gray')
    axes[0,0].set_title('Citra Asli (Noisy)')
    
    axes[0,1].hist(noisy.flatten(), bins=50, color='blue')
    axes[0,1].axvline(x=T_otsu, color='red', linestyle='--', label=f'T_otsu={T_otsu}')
    axes[0,1].legend()
    axes[0,1].set_title('Histogram + Otsu Threshold')
    
    axes[0,2].imshow(binary_otsu, cmap='gray')
    axes[0,2].set_title(f'Otsu (T={T_otsu})\nσ²_B = ω₁ω₂(μ₁-μ₂)²')
    
    axes[1,0].imshow(region, cmap='gray')
    axes[1,1].set_title('Region Growing\n|f-μ| < T')
    
    axes[1,1].imshow(kmeans_result, cmap='jet')
    axes[1,1].set_title('K-Means (K=4)\nmin Σ||x-μ_k||²')

    from skimage.color import label2rgb
    axes[1,2].imshow(label2rgb(kmeans_result))
    axes[1,2].set_title('K-Means (Colored)')
    
    plt.tight_layout()
    plt.savefig('bab8_segmentasi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 8 disimpan")
