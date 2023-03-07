# An Interpretable MRI Reconstruction Network with Two-grid-cycle Correction and Geometric Prior Distillation

This repository contains the CS-MRI reconstruction pytorch codes for the following paper：  
Xiaohong Fan, Yin Yang, Ke Chen, Jianping Zhang*, Ke Dong, “A Unifying Multi-sampling-ratio CS-MRI Framework With Two-grid-cycle Correction and Geometric Prior Distillation”. arXiv, 2022. [[pdf]](https://arxiv.org/abs/2205.07062) 

Abstract: CS is an efficient method to accelerate the acquisition of MR images from under-sampled k-space data. Although existing deep learning CS-MRI methods have achieved considerably impressive performance, explainability and generalizability continue to be challenging for such methods since most of them are not flexible enough to handle multi-sampling-ratio reconstruction assignments, often the transition from mathematical analysis to network design not always natural enough. In this work, to tackle explainability and generalizability, we propose a unifying deep unfolding multi-sampling-ratio CS-MRI framework, by merging advantages of model-based and deep learning-based methods. The combined approach offers more generalizability than previous works whereas deep learning gains explainability through a geometric prior module. Inspired by multigrid algorithm, we first embed the CS-MRI-based optimization algorithm into correction-distillation scheme that consists of three ingredients: pre-relaxation module, correction module and geometric prior distillation module. Furthermore, we employ a condition module to learn adaptively step-length and noise level from compressive sampling ratio in every stage, which enables the proposed framework to jointly train multi-ratio tasks through a single model. The proposed model can not only compensate the lost contextual information of reconstructed image which is refined from low frequency error in geometric characteristic k-space, but also integrate the theoretical guarantee of model-based methods and the superior reconstruction performances of deep learning-based methods. All physical-model parameters are learnable, and numerical experiments show that our framework outperforms state-of-the-art methods in terms of qualitative and quantitative evaluations.

![image](https://user-images.githubusercontent.com/48355877/185527812-9de873ad-f705-4336-8ba4-60068356276d.png)

These codes are built on PyTorch and tested on Ubuntu 18.04/20.04 (Python3.x, PyTorch>=0.4) with Intel Xeon CPU E5-2630 and Nvidia Tesla V100 GPU.

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
