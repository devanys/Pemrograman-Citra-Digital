def bab10_morfologi():
    print("\n" + "="*80)
    print("BAB 10: MORFOLOGI CITRA")
    print("="*80)
    print("\n[10.1] Dilasi dan Erosi")
    print("-" * 40)
    print("Dilasi: A ⊕ B = {z | (B̂)_z ∩ A ≠ ∅}")
    print("Erosi: A ⊖ B = {z | (B)_z ⊆ A}")
    
    def dilasi_manual(citra, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(citra, pad, mode='constant', constant_values=0)
        output = np.zeros_like(citra)
        
        for i in range(citra.shape[0]):
            for j in range(citra.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                output[i, j] = np.max(region * kernel)
        
        return output
    
    def erosi_manual(citra, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(citra, pad, mode='constant', constant_values=255)
        output = np.zeros_like(citra)
        
        for i in range(citra.shape[0]):
            for j in range(citra.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                valid_pixels = region[kernel == 1]
                if len(valid_pixels) > 0:
                    output[i, j] = np.min(valid_pixels)
        
        return output
    
    print("\n[10.2] Opening dan Closing")
    print("-" * 40)
    print("Opening: A ∘ B = (A ⊖ B) ⊕ B")
    print("Closing: A • B = (A ⊕ B) ⊖ B")
    print("\nOpening: menghilangkan objek kecil, memecah objek tipis")
    print("Closing: mengisi lubang kecil, menghubungkan objek terpisah")
    
    kernel_cross = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
    kernel_box = np.ones((3, 3))
    
    citra_binary = np.zeros((200, 200), dtype=np.uint8)
    
    cv2.rectangle(citra_binary, (50, 50), (150, 150), 255, -1)
    
    noise = np.random.random((200, 200)) < 0.02
    citra_binary[noise] = 255
  
    cv2.circle(citra_binary, (100, 100), 15, 0, -1)
    
    dilated = dilasi_manual(citra_binary, kernel_box)
    eroded = erosi_manual(citra_binary, kernel_box)
    
    opened = dilasi_manual(eroded, kernel_box)

    closed = erosi_manual(dilated, kernel_box)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BAB 10: Morfologi Citra', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(citra_binary, cmap='gray')
    axes[0,0].set_title('Citra Binary Asli\n(dengan noise & lubang)')
    
    axes[0,1].imshow(dilated, cmap='gray')
    axes[0,1].set_title('Dilasi A⊕B\nmax{A∩B̂}')
    
    axes[0,2].imshow(eroded, cmap='gray')
    axes[0,2].set_title('Erosi A⊖B\nmin{A∩B}')
    
    axes[1,0].imshow(opened, cmap='gray')
    axes[1,1].set_title('Opening A∘B\n(A⊖B)⊕B - hilangkan noise')
    
    axes[1,1].imshow(closed, cmap='gray')
    axes[1,1].set_title('Closing A•B\n(A⊕B)⊖B - isi lubang')

    gradient = dilated.astype(int) - eroded.astype(int)
    axes[1,2].imshow(gradient, cmap='gray')
    axes[1,2].set_title('Gradient\nA⊕B - A⊖B')
    
    plt.tight_layout()
    plt.savefig('bab10_morfologi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 10 disimpan")
