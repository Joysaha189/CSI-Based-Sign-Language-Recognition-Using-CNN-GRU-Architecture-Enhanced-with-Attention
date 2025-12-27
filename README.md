# CSI-Based-Sign-Language-Recognition-Using-CNN-GRU-Architecture-Enhanced-with-Attention









This repository contains the implementation and experimental analysis of a CSI-based sign language recognition system using a hybrid CNN–GRU architecture enhanced with multi-head attention.



The project was completed for **IECE 566: Deep Learning (Fall 2025)** at the **University at Albany, SUNY**.









###### **📌 Project Overview**







Camera-based sign language recognition systems raise privacy concerns and often require controlled environments. This project explores WiFi **Channel State Information (CSI)** as a device-free, privacy-preserving alternative for recognizing fine-grained hand and arm gestures using deep learning.



CSI captures amplitude and phase variations across subcarriers and antennas, producing high-dimensional spatio-temporal signals. However, CSI data is noisy, user-dependent, and sensitive to environmental changes. This project investigates whether a **CNN–GRU–Attention architecture**, combined with **CSI-specific preprocessing, data augmentation, and self-supervised pretraining**, can achieve robust and generalizable sign language recognition.







**The study includes:**







* A hybrid CNN–GRU–Attention architecture for spatial–temporal modeling
* CSI-specific preprocessing and amplitude–phase fusion
* Extensive comparison with classical machine learning baselines
* Ablation studies to analyze architectural contributions
* Self-supervised pretraining with data augmentation
* Grad-CAM–based explainability for model interpretation









###### 📁 Repository Structure







```



├── Code/



│   ├── simple\_classifiers\_final.py       # Classical ML baselines

│   ├── cnn\_gru\_attention.py              # CNN-GRU-Attention model

│   ├── hyper-opt.py                      # Hyperparameter optimization

│   ├── ablation-study.py                 # Ablation experiments

│   ├── self-supervised-pretraining.py    # Pretraining with augmentation

│   ├── explainability.py                 # Grad-CAM analysis



│



├── Data/



│   └── Home/                            # Subset of the Home CSI dataset (for demonstration)



│



├── Results/



│   ├── plots/                           # All results plot



│



├── Deliverables/



│   ├── DL\\\_Project Report.pdf               # Final project report



│   ├── DL\_Project\\\_Overview.pdf             # Initial project proposal



│   └── Poster\\\_Joy.pdf                      # Project Presentation Slides



│



├── Materials/



│   ├── Sign Detection with CSI               # Sign-Fi dataset reference paper



│   └── FuseLoc.pdf                  # Paper used for phase pre-processing



│



└── README.md



```











###### **🧠 Problem Formulation**







The objective is to learn a mapping from CSI tensors to gesture labels by minimizing a multi-class cross-entropy loss:







$$



\\min\\\_{\\theta} ; L(\\\\theta) = \\frac{1}{N}\\\\sum\\\_{i=1}^{N} \\ell(x\\\_i, y\\\_i; \\theta)



$$














where each CSI sample $x_i$ is a tensor of shape (200 × 60 × 3) capturing temporal evolution, subcarrier variation, and multi-antenna spatial patterns.

![simple](Results/simple-classifiers.png)
CSI data is inherently noisy and high-dimensional, making joint spatial–temporal modeling essential for accurate gesture recognition.








###### ⚙️ Methodology







**Key Components**







* Moving-average filtering and **amplitude–phase fusion**
* CNN blocks with BatchNorm, ReLU, SE blocks, pooling, and dropout
* **Bidirectional GRU** layers for temporal modeling
* **Multi-head attention** to emphasize informative time steps
* Fully connected layers for final classification





**Neural Network Architecture**







![cnn](Results/model.png)





* **Input:** CSI tensor (200 × 60 × 3)
* **CNN** extracts spatial representations
* **GRU** models temporal dependencies
* **Attention** focuses on discriminative segments
* **Softmax** output for gesture classification













📊 Experimental Setup







CSI Datasets







| Dataset | # Signs | Repetitions | # Instances |



| ------- | ------- | ----------- | ----------- |



| Home    | 276     | 10          | 2,760       |



| Lab     | 276     | 20          | 5,520       |



| Lab150  | 150     | 10          | 7,500       |







> ⚠️ Due to size constraints, only a subset of the Home dataset is included in this repository under `Data/`. The Lab150 dataset contains data from five users collected at different times, introducing significant user and environmental variability.















**Training Configuration**







* Batch size: 256
* Optimizer: AdamW
* Weight decay: 1 × 10⁻⁴
* Learning rate: 5 × 10⁻⁴
* Epochs: up to 150 with early stopping
* Cosine annealing learning rate schedule
* Label smoothing (ε = 0.1)
* Gradient clipping (max norm = 1.0)













###### **📈 Results Summary**







* CNN–GRU–Attention significantly outperforms classical ML models
* Achieves 94–95% accuracy on Home and Lab datasets
* Using pretraining with augmentation increases the accuracy to ~99%
* Maintains ~87% accuracy on Lab150 despite high user variability
* Attention provides consistent but marginal performance gains
* Pretraining with augmentation improves convergence stability



![results](Results/baselines-comp.png)







Detailed results for all configurations  are available effect to attention head, ablation study, results using Grad-Cam in the `Results/` directory also in project report in `Deliverables/`.















###### **✅ Conclusions**



* Joint CNN and GRU modeling is essential for CSI-based gesture recognitio
* Attention improves temporal focus and interpretability
* Classical ML models fail to generalize under user variability
* Pretraining and augmentation significantly enhance robustness
* User diversity remains the primary limitation for real-world deployment









###### 📚 References







1.Yongsen Ma, Gang Zhou, Shuangquan Wang, Hongyang Zhao, and Woosub Jung. Signfi: Sign language recognition using wifi. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 2(1):Article 23, 21 pages, March 2018.





2\. T. F. Sanam and H. Godrich, "FuseLoc: A CCA Based Information Fusion for Indoor Localization Using CSI Phase and Amplitude of Wifi Signals," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, UK, 2019, pp. 7565-7569, doi: 10.1109/ICASSP.2019.8683316. 







###### **Project Status**











✅ Completed — Baseline implementation







🔧 Open for enhancements and upgrades























###### **Acknowledgements**











The initial components of this project, including CSI data preprocessing and baseline pipeline development, were carried out during my undergraduate research. Details of that work can be found at [here](https://github.com/Joysaha189/Implementation-Friendly-CNN-For-Sign-Language-Recognition-Using-Wi-Fi-CSI-Data).







I sincerely thank my undergraduate supervisors, Dr. Hafiz Imtiaz and Dr. Tahsina Farah Sanam, for their guidance and foundational contributions, which made this work possible.







The dataset and baseline model architecture are based on the SignFi framework (\[https://yongsen.github.io/SignFi/](https://yongsen.github.io/SignFi/)).







I am also grateful to Dr. Sourabh Sihag for his valuable guidance and feedback throughout the completion of this project as part of IECE 566.



















###### **Author**















**Joy Saha**







University at Albany, SUNY



























###### **License**







This project is for academic and educational purposes.













