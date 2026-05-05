def bab6_domain_frekuensi():
    print("\n" + "="*80)
    print("BAB 6: TRANSFORMASI DOMAIN FREKUENSI")
    print("="*80)
    
    print("\n[6.1] Discrete Fourier Transform (DFT)")
    print("-" * 40)
    print("Persamaan DFT 2D:")
    print("F(u,v) = Σ_x Σ_y f(x,y) · exp(-j2π(ux/M + vy/N))")
    print("\nPersamaan Inverse DFT (IDFT):")
    print("f(x,y) = (1/MN) Σ_u Σ_v F(u,v) · exp(j2π(ux/M + vy/N))")
    
    x = np.linspace(0, 2*np.pi, 256)
    y = np.linspace(0, 2*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    citra_sinyal = ((np.sin(5*X) + np.sin(10*Y) + 1) * 100 + 28).astype(np.uint8)
    
    F = fft2(citra_sinyal.astype(float))
    F_shifted = fftshift(F)
    magnitude = np.log1p(np.abs(F_shifted))
    phase = np.angle(F_shifted)

    print("\n[6.2] Filtering di Domain Frekuensi")
    print("-" * 40)
    print("G(u,v) = H(u,v) · F(u,v)")
    print("di mana H(u,v) = filter transfer function")
    
    def ideal_lowpass(shape, cutoff):
        rows, cols = shape
        center = (rows//2, cols//2)
        Y, X = np.ogrid[:rows, :cols]
        D = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        return D <= cutoff
    
    def gaussian_lowpass(shape, cutoff, sigma=None):
        if sigma is None:
            sigma = cutoff
        rows, cols = shape
        center = (rows//2, cols//2)
        Y, X = np.ogrid[:rows, :cols]
        D_sq = (X - center[1])**2 + (Y - center[0])**2
        return np.exp(-D_sq / (2 * sigma**2))
    
    def butterworth_lowpass(shape, cutoff, order=2):
        rows, cols = shape
        center = (rows//2, cols//2)
        Y, X = np.ogrid[:rows, :cols]
        D = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        D = np.maximum(D, 0.001)  # Avoid division by zero
        return 1 / (1 + (D / cutoff)**(2 * order))

    cutoff = 50
    H_ideal = ideal_lowpass(citra_sinyal.shape, cutoff).astype(float)
    H_gaussian = gaussian_lowpass(citra_sinyal.shape, cutoff)
    H_butterworth = butterworth_lowpass(citra_sinyal.shape, cutoff)

    G_ideal = F_shifted * H_ideal
    G_gaussian = F_shifted * H_gaussian
    G_butterworth = F_shifted * H_butterworth

    def inverse_fft_filtered(G_filtered):
        G_ishifted = ifftshift(G_filtered)
        g = np.real(ifft2(G_ishifted))
        return np.clip(g, 0, 255).astype(np.uint8)
    
    citra_ideal = inverse_fft_filtered(G_ideal)
    citra_gaussian = inverse_fft_filtered(G_gaussian)
    citra_butterworth = inverse_fft_filtered(G_butterworth)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('BAB 6: Transformasi Domain Frekuensi', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_sinyal, cmap='gray')
    axes[0,0].set_title('Citra Asli\n(Sinyal Spasial)')
    
    axes[0,1].imshow(magnitude, cmap='gray')
    axes[0,1].set_title('Magnitude Spectrum\n|F(u,v)|')
    
    axes[0,2].imshow(phase, cmap='hsv')
    axes[0,2].set_title('Phase Spectrum\n∠F(u,v)')
    
    axes[0,3].imshow(H_gaussian, cmap='gray')
    axes[0,3].set_title('Gaussian LPF\nH = exp(-D²/2D₀²)')
    
    axes[1,0].imshow(citra_ideal, cmap='gray')
    axes[1,0].set_title('Ideal LPF\n(Ringing effect)')
    
    axes[1,1].imshow(citra_gaussian, cmap='gray')
    axes[1,1].set_title('Gaussian LPF\n(Smooth)')
    
    axes[1,2].imshow(citra_butterworth, cmap='gray')
    axes[1,2].set_title('Butterworth LPF\nH = 1/(1+(D/D₀)^2n)')

    D = np.linspace(0, 128, 256)
    H_ideal_1d = (D <= cutoff).astype(float)
    H_gauss_1d = np.exp(-D**2 / (2 * cutoff**2))
    H_butter_1d = 1 / (1 + (D / cutoff)**4)
    
    axes[1,3].plot(D, H_ideal_1d, 'b-', label='Ideal')
    axes[1,3].plot(D, H_gauss_1d, 'g-', label='Gaussian')
    axes[1,3].plot(D, H_butter_1d, 'r-', label='Butterworth')
    axes[1,3].axvline(x=cutoff, color='k', linestyle='--', label=f'D₀={cutoff}')
    axes[1,3].set_xlabel('D(u,v)')
    axes[1,3].set_ylabel('H(u,v)')
    axes[1,3].set_title('Profil Filter')
    axes[1,3].legend()
    axes[1,3].grid(True)
    
    plt.tight_layout()
    plt.savefig('bab6_frekuensi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 6 disimpan")
