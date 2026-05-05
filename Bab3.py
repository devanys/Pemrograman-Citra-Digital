def bab3_model_warna():
    print("\n" + "="*80)
    print("BAB 3: MODEL WARNA DAN TRANSFORMASI WARNA")
    print("="*80)
    
    print("\n[3.1] Model Warna")
    print("-" * 40)
    print("RGB: f(x,y) = [R, G, B]ᵀ, masing-masing ∈ [0, 255]")
    print("HSV: f(x,y) = [H, S, V]ᵀ, H ∈ [0, 360), S,V ∈ [0, 1]")
    print("LAB: f(x,y) = [L*, a*, b*]ᵀ, L* ∈ [0, 100]")
  
    citra_warna = np.zeros((200, 200, 3), dtype=np.uint8)
    
    for i in range(200):
        for j in range(200):
            citra_warna[i, j, 0] = int(255 * j / 200)  # R
            citra_warna[i, j, 1] = int(255 * i / 200)  # G
            citra_warna[i, j, 2] = int(255 * (1 - i/200))  # B
    
    hsv = rgb2hsv(citra_warna)
    lab = rgb2lab(citra_warna)
    gray = rgb2gray(citra_warna)
    
    def rgb_to_grayscale_manual(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.uint8)
    
    gray_manual = rgb_to_grayscale_manual(citra_warna)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 3: Model Warna dan Transformasi', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_warna)
    axes[0,0].set_title('RGB Space')
    
    axes[0,1].imshow(hsv[:,:,0], cmap='hsv')
    axes[0,1].set_title('HSV - Hue')
    
    axes[0,2].imshow(hsv[:,:,1], cmap='gray')
    axes[0,2].set_title('HSV - Saturation')
    
    axes[1,0].imshow(lab[:,:,0], cmap='gray')
    axes[1,0].set_title('LAB - L* (Lightness)')
    
    axes[1,1].imshow(gray, cmap='gray')
    axes[1,1].set_title('Grayscale (skimage)')
    
    axes[1,2].imshow(gray_manual, cmap='gray')
    axes[1,2].set_title('Grayscale (Manual)\nY = 0.299R + 0.587G + 0.114B')
    
    plt.tight_layout()
    plt.savefig('bab3_model_warna.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 3 disimpan")

