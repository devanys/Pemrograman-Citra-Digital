def bab7_restorasi():
    print("\n" + "="*80)
    print("BAB 7: RESTORASI DAN REKONSTRUKSI CITRA")
    print("="*80)
  
    print("\n[7.1] Model Degradasi Citra")
    print("-" * 40)
    print("Persamaan degradasi:")
    print("g(x,y) = h(x,y) * f(x,y) + n(x,y)")
    print("di domain frekuensi:")
    print("G(u,v) = H(u,v) · F(u,v) + N(u,v)")
    print("  h = PSF (Point Spread Function)")
    print("  n = noise")
    
    print("\n[7.2] Inverse Filtering")
    print("-" * 40)
    print("F_est(u,v) = G(u,v) / H(u,v)")
    print("Masalah: pembesaran noise saat H(u,v) ≈ 0")
    
    def inverse_filter(G, H, threshold=0.01):
        H_safe = np.where(np.abs(H) > threshold, H, threshold)
        F_est = G / H_safe
        return F_est
    
    print("\n[7.2] Wiener Filtering")
    print("-" * 40)
    print("Persamaan Wiener:")
    print("F_est(u,v) = [H*(u,v) / (|H(u,v)|² + S_n(u,v)/S_f(u,v))] · G(u,v)")
    print("\nVersi sederhana (rasio SNR konstan):")
    print("F_est(u,v) = [H*(u,v) / (|H(u,v)|² + K)] · G(u,v)")
    print("  H* = complex conjugate dari H")
    print("  K = rasio noise-to-signal power")
    
    def wiener_filter(G, H, K=0.01):
        H_conj = np.conj(H)
        H_power = np.abs(H)**2
        W = H_conj / (H_power + K)
        F_est = W * G
        return F_est
    )
    def motion_blur_psf(size, length, angle):
        """Membuat PSF motion blur"""
        psf = np.zeros((size, size))
        center = size // 2
        angle_rad = np.deg2rad(angle)
        
        for i in range(-length//2, length//2 + 1):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1
        
        return psf / psf.sum()

    citra_test = np.zeros((256, 256), dtype=np.uint8)
    cv2.putText(citra_test, 'IMAGE', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    psf = motion_blur_psf(15, 10, 0)
    psf_padded = np.zeros(citra_test.shape)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    
    H = fftshift(fft2(ifftshift(psf_padded)))
    F = fft2(citra_test.astype(float))

    noise_power = 100
    noise = np.random.normal(0, np.sqrt(noise_power), citra_test.shape)
    G = H * F + noise

    F_inverse = inverse_filter(G, H, threshold=0.1)
    citra_inverse = np.clip(np.real(ifft2(F_inverse)), 0, 255).astype(np.uint8)
    
    F_wiener = wiener_filter(G, H, K=0.01)
    citra_wiener = np.clip(np.real(ifft2(F_wiener)), 0, 255).astype(np.uint8)
    
    citra_degraded = np.clip(np.real(ifft2(G)), 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 7: Restorasi Citra', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_test, cmap='gray')
    axes[0,0].set_title('Citra Asli f(x,y)')
    
    axes[0,1].imshow(np.abs(psf), cmap='gray')
    axes[0,1].set_title('PSF h(x,y)\n(Motion Blur)')
    
    axes[0,2].imshow(citra_degraded, cmap='gray')
    axes[0,2].set_title('Citra Terdegradasi\ng = h*f + n')
    
    axes[1,0].imshow(np.log1p(np.abs(H)), cmap='gray')
    axes[1,0].set_title('H(u,v) Transfer Function')
    
    axes[1,1].imshow(citra_inverse, cmap='gray')
    axes[1,1].set_title('Inverse Filter\nF = G/H (noise amplified)')
    
    axes[1,2].imshow(citra_wiener, cmap='gray')
    axes[1,2].set_title('Wiener Filter\nF = [H*/(|H|²+K)]·G')
    
    plt.tight_layout()
    plt.savefig('bab7_restorasi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 7 disimpan")
