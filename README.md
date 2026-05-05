<div align="center">

# 📚 Pemrograman Citra Digital
### Teori, Implementasi, dan Aplikasi Berbasis Deep Learning

<img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/NumPy-Math-orange?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/OpenCV-Vision-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Scikit--Image-Processing-yellow?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Image">

<br><br>
<i>Output Repositori Code</i>
<br>
<b>Oleh: Devan Yusfa Sukmadya</b>
<br>

</div>

---

## 📖 About
Implementasi Python yang secara eksplisit menerjemahkan persamaan matematika pengolahan citra digital dan deep learning. Setiap modul dilengkapi  teori (kalkulus, aljabar linear, deret Fourier)

---

## 📸 Visualisasi & Implementasi Per Bab

### BAB 1: Konsep Dasar dan Sistem Pengolahan Citra
<p align="center">
  <img width="800" alt="bab1_konsep_dasar" src="https://github.com/user-attachments/assets/bfc1229c-aaba-484e-a656-0a14f5147d49" />
</p>
<p><i>Memulai dari representasi matriks hingga implementasi manual persamaan SSIM, Entropi Shannon, dan Teorema Nyquist-Shannon.</i></p>

### BAB 2: Akuisisi dan Representasi Citra
<p align="center">
  <img width="800" alt="bab2_sampling_kuantisasi" src="https://github.com/user-attachments/assets/c402799d-a0a3-4420-9248-a07fe398aa20" />
  <br>
  <img width="800" alt="bab2_noise" src="https://github.com/user-attachments/assets/00a1d7b3-b617-4c15-8848-3c9569be558e" />
</p>
<p><i>Simulasi digitalisasi (sampling & kuantisasi), pemodelan noise (Gaussian, Salt & Pepper), dan perhitungan PSNR.</i></p>

### BAB 3: Model Warna dan Transformasi Warna
<p align="center">
  <img width="800" alt="bab3_model_warna" src="https://github.com/user-attachments/assets/e5c1cc75-9e2b-4d51-94dd-c23e6a8505c9" />
</p>
<p><i>Implementasi transformasi antar ruang warna (RGB, HSV, LAB) dan penerapan persamaan luminance ITU-R BT.601.</i></p>

### BAB 4: Peningkatan Kualitas Citra (Image Enhancement)
<p align="center">
  <img width="800" alt="bab4_enhancement" src="https://github.com/user-attachments/assets/9cb79865-39f6-4a69-9db1-4ebae0c1ccd8" />
</p>
<p><i>Penerapan transformasi logaritmik, Gamma Correction ($s = c \cdot r^\gamma$), dan pembuatan Histogram Equalization berbasis CDF.</i></p>

### BAB 5: Operasi Spasial dan Filtering
<p align="center">
  <img width="800" alt="bab5_filtering" src="https://github.com/user-attachments/assets/fef2e09b-e8ab-4e17-9e05-6a056d3cc4b2" />
</p>
<p><i>Konvolusi 2D dari nol, diterapkan pada filter Mean, distribusi normal Gaussian, Median, dan operator Laplacian ($\nabla^2f$).</i></p>

### BAB 6: Transformasi Domain Frekuensi
<p align="center">
  <img width="800" alt="bab6_frekuensi" src="https://github.com/user-attachments/assets/df6d209e-994a-45c4-9dd5-ab752f1365a0" />
</p>
<p><i>Analisis Magnitude/Phase menggunakan DFT/FFT dan implementasi persamaan filter Lowpass (Ideal, Gaussian, Butterworth).</i></p>

### BAB 7: Restorasi dan Rekonstruksi Citra
<p align="center">
  <img width="800" alt="bab7_restorasi" src="https://github.com/user-attachments/assets/bdecc315-5530-4a5b-ae8e-091d9264f5da" />
</p>
<p><i>Restorasi citra terdegradasi menggunakan pendekatan matematis Inverse Filtering dan Wiener Filtering.</i></p>

### BAB 8: Segmentasi Citra
<p align="center">
  <img width="800" alt="bab8_segmentasi" src="https://github.com/user-attachments/assets/8e4f94be-b164-4dcb-8b60-3565e12cd406" />
</p>
<p><i>Partisi citra menggunakan Otsu (memaksimalkan *between-class variance*), Region Growing, dan K-Means Clustering.</i></p>

### BAB 9: Deteksi Tepi dan Kontur
<p align="center">
  <img width="800" alt="bab9_deteksi_tepi" src="https://github.com/user-attachments/assets/8ddc3e8e-c977-47d8-8af1-8a328ab088ee" />
</p>
<p><i>Ekstraksi batas objek melalui operasi turunan parsial (Sobel, Prewitt) dan algoritma multi-tahap Canny Edge Detector.</i></p>

### BAB 10: Morfologi Citra
<p align="center">
  <img width="800" alt="bab10_morfologi" src="https://github.com/user-attachments/assets/cd613036-0b87-42e2-86ec-a14e7ba41cdf" />
</p>
<p><i>Manipulasi struktur citra biner menggunakan operasi Dilasi ($\max$), Erosi ($\min$), Opening, dan Closing.</i></p>

### BAB 11: Ekstraksi Fitur dan Representasi Citra
<p align="center">
  <img width="800" alt="bab11_ekstraksi_fitur" src="https://github.com/user-attachments/assets/2f575adb-bbf8-4eca-8b76-7174a8ff927f" />
</p>
<p><i>Ekstraksi fitur diskriptif tekstur (GLCM, LBP) dan reduksi dimensi menggunakan dekomposisi eigen pada PCA.</i></p>

### BAB 12: Pengenalan Pola dan Machine Learning
<p align="center">
  <img width="800" alt="bab12_ml" src="https://github.com/user-attachments/assets/b8744e50-9b06-40f1-9141-176234fe9bd7" />
</p>
<p><i>Visualisasi *decision boundary* algoritma klasik (KNN, SVM, Decision Tree) dan evaluasi kurva ROC-AUC.</i></p>

### BAB 13: Deep Learning untuk Pengolahan Citra
<p align="center">
  <img width="800" alt="bab13_deep_learning" src="https://github.com/user-attachments/assets/9cdb2d1b-f641-435c-9c50-7690d89dc01f" />
</p>
<p><i>Fondasi arsitektur jaringan saraf: fungsi aktivasi, persamaan dimensi konvolusi, dan simulasi *skip connection* ResNet.</i></p>

### BAB 14: Generative AI dan Aplikasi Lanjut
<p align="center">
  <img width="800" alt="bab14_generative_ai" src="https://github.com/user-attachments/assets/2e3c7f89-86de-46ad-a6a4-b69fd92c7837" />
</p>
<p><i>Simulasi *Forward Process* Diffusion Model, arsitektur GAN, dan penerapan Grad-CAM untuk *Explainable AI*.</i></p>

---

## 📚 Sumber Referensi

[1] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York: Pearson, 2018.  
[2] B. Jähne, *Digital Image Processing*, 6th ed. Berlin: Springer-Verlag, 2005.  
[3] A. A. A. Setio et al., "Pulmonary nodule detection in CT images," *IEEE Trans. Medical Imaging*, vol. 35, no. 5, pp. 1160–1169, 2016.  
[4] V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy," *JAMA*, vol. 316, no. 22, pp. 2402–2410, 2016.  
[5] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017.  
[6] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? The KITTI vision benchmark suite," *Proc. IEEE CVPR*, 2012, pp. 3354–3361.  
[7] S. Kamilaris and F. X. Prenafeta-Boldu, "Deep learning in agriculture: A survey," *Computers and Electronics in Agriculture*, vol. 147, pp. 70–90, 2018.  
[8] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.  
[9] E. Ells et al., "Telemedicine approach to screening for severe retinopathy of prematurity," *Ophthalmology*, vol. 110, no. 11, pp. 2113–2117, 2003.  
[10] E. P. Simoncelli and B. A. Olshausen, "Natural image statistics and neural representation," *Annual Review of Neuroscience*, vol. 24, pp. 1193–1216, 2001.  
[11] W. K. Pratt, *Digital Image Processing: PIKS Scientific Inside*, 4th ed. New York: Wiley-Interscience, 2007.  
[12] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Trans. Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.  
[13] C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, 1948.  
[14] D. H. Hubel and T. N. Wiesel, "Receptive fields and functional architecture of monkey striate cortex," *Journal of Physiology*, vol. 195, no. 1, pp. 215–243, 1968.  
[15] C. Shao et al., "Remote sensing image super-resolution using sparse representation," *IEEE Trans. Geoscience and Remote Sensing*, vol. 57, pp. 1529–1542, 2019.  
[16] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *Journal of Big Data*, vol. 6, no. 60, pp. 1–48, 2019.  
[17] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," *Proc. IEEE CVPR*, 2015, pp. 3431–3440.  
[18] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," *Proc. MICCAI*, 2015, pp. 234–241.  
[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," *Proc. NeurIPS*, 2012, pp. 1097–1105.  
[20] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv:1711.05225*, 2017.  
[21] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115–118, 2017.  
[22] E. J. Topol, "High-performance medicine: The convergence of human and artificial intelligence," *Nature Medicine*, vol. 25, pp. 44–56, 2019.  
[23] H. Kolb, E. Fernandez, and R. Nelson, *Webvision: The Organization of the Retina and Visual System*. Salt Lake City: University of Utah, 1995.  
[24] T. Y. Wong et al., "The eye as a window to the soul of the heart," *European Heart Journal*, vol. 28, no. 11, pp. 1310–1313, 2007.  
[25] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," *Proc. IEEE CVPR*, 2016, pp. 770–778.  
[26] H. Moravec, *Mind Children: The Future of Robot and Human Intelligence*. Cambridge: Harvard University Press, 1988.  
[27] A. Kirillov et al., "Segment Anything," *Proc. IEEE/CVF ICCV*, 2023, pp. 4015–4026.  
[28] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal approximators," *Neural Networks*, vol. 2, no. 5, pp. 359–366, 1989.  
[29] Z. Obermeyer and E. J. Emanuel, "Predicting the future — big data, machine learning, and clinical medicine," *NEJM*, vol. 375, no. 13, pp. 1216–1219, 2016.  
[30] World Health Organization, *World Report on Vision*. Geneva: WHO, 2019.  
[31] Y. Teo et al., "Global prevalence of diabetic retinopathy," *Ophthalmology*, vol. 128, no. 11, pp. 1580–1591, 2021.  
[32] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," *Proc. IEEE ICCV*, 2017, pp. 618–626.  
[33] I. J. Goodfellow et al., "Generative adversarial nets," *Proc. NeurIPS*, 2014, pp. 2672–2680.  
[34] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," *Proc. NeurIPS*, 2020.  
[35] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," *Proc. ICML*, 2019, pp. 6105–6114.  
[36] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *Proc. ICLR*, 2021.  
[37] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," *Proc. ICML*, 2015, pp. 448–456.  
[38] H. Zhang et al., "Mixup: Beyond empirical risk minimization," *Proc. ICLR*, 2018.  
[39] F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, vol. 18, pp. 203–211, 2021.  
[40] C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," *Proc. IEEE ICCV*, 1998, pp. 839–846.  
