def bab9_deteksi_tepi():
    print("\n" + "="*80)
    print("BAB 9: DETEKSI TEPI DAN KONTUR")
    print("="*80)
  
    print("\n[9.1] Konsep Tepi dan Gradien")
    print("-" * 40)
    print("Gradien citra:")
    print("∇f = [∂f/∂x, ∂f/∂y]ᵀ")
    print("Magnitudo gradien:")
    print("|∇f| = √[(∂f/∂x)² + (∂f/∂y)²]")
    print("Arah gradien:")
    print("θ = atan2(∂f/∂y, ∂f/∂x)")
   
    print("\n[9.2] Operator Sobel")
    print("-" * 40)
    print("Gx = [[-1, 0, 1],\n       [-2, 0, 2],\n       [-1, 0, 1]]")
    print("Gy = [[-1, -2, -1],\n       [ 0,  0,  0],\n       [ 1,  2,  1]]")
    print("|G| = √(Gx² + Gy²)")
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]])
    
    def konvolusi(citra, kernel):
        from scipy.signal import convolve2d
        return convolve2d(citra, kernel, mode='same', boundary='symm')
    
    def deteksi_tepi_sobel(citra):
        gx = konvolusi(citra.astype(float), sobel_x)
        gy = konvolusi(citra.astype(float), sobel_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        return magnitude, direction, gx, gy
    
    def deteksi_tepi_prewitt(citra):
        """Deteksi tepi Prewitt"""
        gx = konvolusi(citra.astype(float), prewitt_x)
        gy = konvolusi(citra.astype(float), prewitt_y)
        magnitude = np.sqrt(gx**2 + gy**2)
        return magnitude
  
    print("\n[9.2] Canny Edge Detector")
    print("-" * 40)
    print("Langkah:")
    print("1. Gaussian smoothing")
    print("2. Gradien (Sobel)")
    print("3. Non-maximum suppression")
    print("4. Double thresholding (T_low, T_high)")
    print("5. Edge linking by hysteresis")
    
    citra_test = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(citra_test, (50, 50), (200, 200), 200, -1)
    cv2.circle(citra_test, (125, 125), 60, 100, -1)
    cv2.line(citra_test, (30, 30), (230, 230), 150, 2)
    
    mag_sobel, dir_sobel, gx, gy = deteksi_tepi_sobel(citra_test)
    mag_prewitt = deteksi_tepi_prewitt(citra_test)
    canny = feature.canny(citra_test, sigma=1, low_threshold=50, high_threshold=100)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 9: Deteksi Tepi dan Kontur', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_test, cmap='gray')
    axes[0,0].set_title('Citra Asli')
    
    axes[0,1].imshow(mag_sobel, cmap='gray')
    axes[0,1].set_title('Sobel\n|G| = √(Gx² + Gy²)')
    
    axes[0,2].imshow(mag_prewitt, cmap='gray')
    axes[0,2].set_title('Prewitt')
    
    axes[1,0].imshow(dir_sobel, cmap='hsv')
    axes[1,0].set_title('Arah Gradien\nθ = atan2(Gy, Gx)')
    
    axes[1,1].imshow(canny, cmap='gray')
    axes[1,1].set_title('Canny Edge\n(NMS + Hysteresis)')

    contours, _ = cv2.findContours(citra_test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.drawContours(np.zeros_like(citra_test), contours, -1, 255, 2)
    axes[1,2].imshow(contour_img, cmap='gray')
    axes[1,2].set_title(f'Contour Detection\n({len(contours)} contours)')
    
    plt.tight_layout()
    plt.savefig('bab9_deteksi_tepi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 9 disimpan")

