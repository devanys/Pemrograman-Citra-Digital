def bab13_deep_learning():
    print("\n" + "="*80)
    print("BAB 13: DEEP LEARNING UNTUK PENGOLAHAN CITRA")
    print("="*80)
    print("\n[13.1] Neural Network Dasar")
    print("-" * 40)
    print("Persamaan neuron:")
    print("z = w·x + b")
    print("a = σ(z)  (fungsi aktivasi)")
    print("\nFungsi aktivasi:")
    print("  Sigmoid: σ(z) = 1 / (1 + e^(-z))")
    print("  ReLU:    f(z) = max(0, z)")
    print("  Softmax: σ(z_i) = e^{z_i} / Σ_j e^{z_j}")
    
    def sigmoid(z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-z))
    
    def relu(z):
        """ReLU activation"""
        return np.maximum(0, z)
    
    def softmax(z):
        """Softmax activation"""
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    print("\n[13.2] Convolutional Neural Network (CNN)")
    print("-" * 40)
    print("Persamaan konvolusi CNN:")
    print("y[i,j] = Σ_m Σ_n x[i+m, j+n] · w[m,n] + b")
    print("\nOutput size after convolution:")
    print("O = (I - K + 2P) / S + 1")
    print("  I = input size, K = kernel size")
    print("  P = padding, S = stride")
    
    def conv2d_output_size(input_size, kernel_size, padding=0, stride=1):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    I, K, P, S = 224, 3, 1, 1
    O = conv2d_output_size(I, K, P, S)
    print(f"\nContoh: Input={I}, Kernel={K}, Padding={P}, Stride={S}")
    print(f"Output size: {O}")
  
    print("\n[13.3] Arsitektur Modern")
    print("-" * 40)
    print("ResNet - Residual Block:")
    print("y = F(x) + x")
    print("F(x) = lapisan konvolusi")
    print("\nIni menyelesaikan vanishing gradient problem")
    
    def residual_block(x, filters):

        shortcut = x
        
        # F(x)
        F = np.random.randn(*x.shape) * 0.01  # Simulasi conv layer
        F = np.maximum(0, F)  # ReLU
        
        # Add
        y = F + shortcut
        y = np.maximum(0, y)  # ReLU
        
        return y
    
    print("\n[13.4] Regularisasi")
    print("-" * 40)
    print("L2 Regularization (Weight Decay):")
    print("L = L_data + λ Σ |w_i|²")
    print("\nDropout:")
    print("z_i = r_i · a_i, r_i ~ Bernoulli(p)")
    print("E[z] = p · a")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BAB 13: Deep Learning untuk Pengolahan Citra', fontsize=14, fontweight='bold')

    z = np.linspace(-5, 5, 100)
    axes[0,0].plot(z, sigmoid(z), 'b-', linewidth=2, label='Sigmoid')
    axes[0,0].plot(z, relu(z), 'r-', linewidth=2, label='ReLU')
    axes[0,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0,0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    axes[0,0].legend()
    axes[0,0].set_xlabel('z')
    axes[0,0].set_ylabel('σ(z)')
    axes[0,0].set_title('Fungsi Aktivasi')
    axes[0,0].grid(True)

    ax = axes[0,1]
    layers = ['Input\n224×224×3', 'Conv1\n112×112×64', 'Pool\n56×56×64', 
              'Conv2\n56×56×128', 'Pool\n28×28×128', 'FC\n512', 'Output\n10']
    colors = ['gray', 'blue', 'green', 'blue', 'green', 'orange', 'red']
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        ax.add_patch(plt.Rectangle((i, 0.3), 0.8, 0.4, facecolor=color, alpha=0.7))
        ax.text(i + 0.4, 0.5, layer, ha='center', va='center', fontsize=7)
        if i < len(layers) - 1:
            ax.annotate('', xy=(i + 1, 0.5), xytext=(i + 0.8, 0.5),
                       arrowprops=dict(arrowstyle='->', color='black'))
    
    ax.set_xlim(-0.5, len(layers))
    ax.set_ylim(0, 1)
    ax.set_title('Arsitektur CNN Sederhana')
    ax.axis('off')

    ax = axes[1,0]
    ax.annotate('', xy=(1.5, 0.7), xytext=(0, 0.7),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.add_patch(plt.Rectangle((0, 0.5), 1.5, 0.4, facecolor='lightblue', alpha=0.7))
    ax.text(0.75, 0.7, 'Conv + ReLU', ha='center', va='center')
    
    ax.annotate('', xy=(3, 0.7), xytext=(1.5, 0.7),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.add_patch(plt.Rectangle((1.5, 0.5), 1.5, 0.4, facecolor='lightblue', alpha=0.7))
    ax.text(2.25, 0.7, 'Conv + ReLU', ha='center', va='center')

    ax.annotate('', xy=(3, 0.2), xytext=(0, 0.2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.plot([0, 0], [0.2, 0], 'g-', lw=2)
    ax.plot([0, 3], [0, 0], 'g-', lw=2)
    ax.plot([3, 3], [0, 0.2], 'g-', lw=2)
    ax.text(1.5, -0.1, 'Shortcut (Identity)', ha='center', va='center', color='green')

    ax.plot([3, 3.5], [0.2, 0.45], 'k-', lw=2)
    ax.plot([3, 3.5], [0.7, 0.45], 'k-', lw=2)
    ax.add_patch(plt.Circle((3.5, 0.45), 0.1, facecolor='yellow', edgecolor='black'))
    ax.text(3.5, 0.45, '+', ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(4.2, 0.45), xytext=(3.6, 0.45),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(4.4, 0.45, 'ReLU', ha='left', va='center')
    
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_title('Residual Block: y = F(x) + x')
    ax.axis('off')

    ax = axes[1,1]
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    loss_plain = np.log(1 + np.exp(-W1)) + np.log(1 + np.exp(W2))
    loss_resnet = loss_plain * 0.3 
    
    ax.contour(W1, W2, loss_plain, levels=20, cmap='Reds', alpha=0.7)
    ax.contour(W1, W2, loss_resnet, levels=20, cmap='Blues', alpha=0.5)
    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_title('Loss Landscape\n(Merah: Plain, Biru: ResNet)')
    
    plt.tight_layout()
    plt.savefig('bab13_deep_learning.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 13 disimpan")

