def bab12_machine_learning():
    print("\n" + "="*80)
    print("BAB 12: PENGENALAN POLA DAN MACHINE LEARNING")
    print("="*80)
    print("\n[12.1] Konsep Pattern Recognition")
    print("-" * 40)
    print("Klasifikasi: y = f(x)")
    print("  x = vektor fitur input")
    print("  y = label kelas output")
    print("\n[12.2] Algoritma Klasifikasi")
    print("-" * 40)

    print("K-Nearest Neighbors (KNN):")
    print("  ŷ = majority{y_i | i ∈ KNN(x)}")
    print("  Jarak: d(x, x') = ||x - x'||₂")

    print("\nSupport Vector Machine (SVM):")
    print("  min ||w||²/2 + C Σ ξ_i")
    print("  s.t. y_i(w·x_i + b) ≥ 1 - ξ_i")

    print("\nDecision Tree:")
    print("  Split criterion: Gini = 1 - Σ p_k²")
    print("  atau: Entropy = -Σ p_k log(p_k)")
    
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(kernel='rbf', probability=True),
        'Decision Tree': DecisionTreeClassifier(max_depth=5)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    print("\n[12.3] Evaluasi Model")
    print("-" * 40)
    print("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    print("Precision = TP / (TP + FP)")
    print("Recall (Sensitivity) = TP / (TP + FN)")
    print("F1-Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print("AUC = Area Under ROC Curve")

    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 56)
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:<12.4f} {res['precision']:<12.4f} {res['recall']:<12.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('BAB 12: Machine Learning untuk Pengenalan Pola', fontsize=14, fontweight='bold')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx // 2, idx % 2]
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
        ax.set_title(f'{name}\nAcc={results[name]["accuracy"]:.3f}')
        ax.set_xlabel('Fitur 1')
        ax.set_ylabel('Fitur 2')

    ax = axes[1, 1]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('bab12_ml.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Visualisasi BAB 12 disimpan")

