def bab14_generative_ai():
    print("\n" + "="*80)
    print("BAB 14: GENERATIVE AI DAN APLIKASI LANJUT")
    print("="*80)
    print("\n[14.1] Generative Adversarial Network (GAN)")
    print("-" * 40)
    print("Min-Max Game:")
    print("min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]")
    print("  D = Discriminator, G = Generator")
    print("  x = data asli, z = noise input")

    print("\n[14.1] Diffusion Model")
    print("-" * 40)
    print("Forward process (q):")
    print("q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)")
    print("\nReverse process (p):")
    print("p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))")
    print("\nTraining objective:")
    print("L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]")
    
    def diffusion_forward(x_0, t, beta, seed=42):
        np.random.seed(seed)
        alpha = 1 - beta
        alpha_bar = np.cumprod(alpha)
        
        sqrt_alpha_bar = np.sqrt(alpha_bar[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar[t])
        
        epsilon = np.random.randn(*x_0.shape)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
        
        return x_t, epsilon

    x_0 = np.random.randn(64, 64)  
    betas = np.linspace(0.0001, 0.02, 1000) 
    
    print("\n[14.3] Grad-CAM (Gradient-weighted Class Activation Mapping)")
    print("-" * 40)
    print("Persamaan Grad-CAM:")
    print("L_{Grad-CAM}^c = ReLU(Σ_k α_k^c · A^k)")
    print("di mana:")
    print("  α_k^c = (1/Z) Σ_i Σ_j (∂y^c/∂A_{ij}^k)")
    print("  A^k = feature map ke-k")
    print("  y^c = score kelas c")
    
    def grad_cam_simulation(feature_maps, weights, target_class=0):
   
        alpha_k = np.mean(weights, axis=(1, 2)) 
        
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for k in range(feature_maps.shape[0]):
            cam += alpha_k[k] * feature_maps[k]
   
        cam = np.maximum(cam, 0)
     
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
   
    num_filters = 16
    feature_maps = np.random.randn(num_filters, 28, 28)
    weights = np.random.randn(num_filters, 28, 28)
    
    cam = grad_cam_simulation(feature_maps, weights)
    
    print("\n[14.4] Metrik Evaluasi Model Medis")
    print("-" * 40)
    print("Sensitivity (Recall) = TP / (TP + FN)")
    print("Specificity = TN / (TN + FP)")
    print("F1-Score = 2PR / (P + R)")
    print("AUC-ROC = Area Under ROC Curve")
    print("AUC-PR = Area Under Precision-Recall Curve")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('BAB 14: Generative AI dan Explainability', fontsize=14, fontweight='bold')

    axes[0,0].imshow(x_0, cmap='gray')
    axes[0,0].set_title('x₀ (Data Asli)')
    
    for idx, t in enumerate([100, 300, 500, 800]):
        x_t, _ = diffusion_forward(x_0, t, betas, seed=idx)
        ax = axes[0, (idx % 3) + 1] if idx < 2 else axes[1, (idx % 2)]
        ax.imshow(x_t, cmap='gray')
        ax.set_title(f'x_{t}\n(β={betas[t]:.4f})')

    ax = axes[1, 2]
    ax.annotate('', xy=(2, 0.8), xytext=(0, 0.8),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.add_patch(plt.Rectangle((0, 0.6), 2, 0.4, facecolor='lightblue', alpha=0.7))
    ax.text(1, 0.8, 'Generator G(z)', ha='center', va='center')
    
    ax.annotate('', xy=(4, 0.8), xytext=(2, 0.8),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.add_patch(plt.Rectangle((2, 0.6), 2, 0.4, facecolor='lightyellow', alpha=0.7))
    ax.text(3, 0.8, 'Discriminator D(x)', ha='center', va='center')
    
    ax.annotate('', xy=(2, 0.6), xytext=(0, 0.2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 0.1, 'Real Data x', color='green')
    
    ax.annotate('', xy=(2, 0.6), xytext=(-0.5, 0.8),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(-0.5, 0.5, 'Noise z', color='red')
    
    ax.annotate('Real/Fake', xy=(4.5, 0.8), xytext=(4.2, 0.8),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 1.2)
    ax.set_title('GAN Architecture')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bab14_generative_ai.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 14 disimpan")
