# An Interpretable MRI Reconstruction Network with Two-grid-cycle Correction and Geometric Prior Distillation

This repository contains the CS-MRI reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Jianping Zhang*, Ke Dong, “An Interpretable MRI Reconstruction Network with Two-grid-cycle Correction and Geometric Prior Distillation”. arXiv, 2022. [[pdf]](https://arxiv.org/abs/2205.07062)  (Accepted to Biomedical Signal Processing and Control, March, 2023.)

Abstract: Although existing deep learning compressed-sensing-based Magnetic Resonance Imaging (CS-MRI) methods have achieved considerably impressive performance, explainability and generalizability continue to be challenging for such methods since the transition from mathematical analysis to network design not always natural enough, often most of them are not flexible enough to handle multi-sampling-ratio reconstruction assignments. {In this work, to tackle explainability and generalizability, we propose a unifying deep unfolding multi-sampling-ratio interpretable CS-MRI framework.} The combined approach offers more generalizability than previous works whereas deep learning gains explainability through a geometric prior module. Inspired by the multigrid algorithm, we first embed the CS-MRI-based optimization algorithm into correction-distillation scheme that consists of three ingredients: pre-relaxation module, correction module and geometric prior distillation module. Furthermore, we employ a condition module to learn adaptively step-length and noise level, which enables the proposed framework to jointly train multi-ratio tasks through a single model. { The proposed model not only compensates for the lost contextual information of reconstructed image which is refined from low frequency error in geometric characteristic k-space}, but also integrates the theoretical guarantee of model-based methods and the superior reconstruction performances of deep learning-based methods. Therefore, it can give us a novel perspective to design biomedical imaging networks. { Numerical experiments show that our framework outperforms state-of-the-art methods in terms of qualitative and quantitative evaluations.} {Our method achieves 3.18 dB improvement at low CS ratio 10\% and average 1.42 dB improvement over other comparison methods on brain dataset using Cartesian sampling mask.


![image](https://user-images.githubusercontent.com/48355877/185527812-9de873ad-f705-4336-8ba4-60068356276d.png)

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

### Environment  
```
pytorch <= 1.7.1 (recommend 1.6.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.16.1, 0.16.2)

### Citation  
If you find the code helpful in your resarch or work, please cite the following papers. 
```
@Article{Fan2021,
  author    = {Xiaohong Fan and Yin Yang and Ke Chen and Jianping Zhang and Ke Dong},
  journal   = {arXiv},
  title     = {A Unifying Multi-sampling-ratio CS-MRI Framework With Two-grid-cycle Correction and Geometric Prior Distillation},
  year      = {2022},
}
```

### Acknowledgements  
Thanks to the authors of ISTA-Net+ and FISTA-Net, our codes are adapted from the open source codes of ISTA-Net + and FISTA-Net.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@smail.xtu.edu.cn
