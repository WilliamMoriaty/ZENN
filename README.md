# ZENN
 Zentropy-enhanced neural network

<img width="416" alt="image" src="https://github.com/user-attachments/assets/2d0a4fb6-dfdc-4c33-8b7c-9cbd514b94e1" />

Schematic of ZENN and its applications in different areas. Zentropy theory integrates statistical mechanics and quantum mechanics by assigning intrinsic entropy to each system component, thereby capturing internal disparities. By embedding zentropy theory into deep learning as a backward modeling framework, ZENN replaces the internal energy $E^{(k)}$ and $S^{(k)}$ of each configuration with simple neural networks, and integrates information across all configurations through the total free energy F. In this paper, ZENN has been applied in three representative tasks—multi-source data integration, energy landscape reconstruction, and inference of Fe₃Pt alloy properties—demonstrating its potential as a powerful framework that effectively bridges statistical mechanics and machine learning. 
# Results on image and text classification tasks
All simulations were performed on a server equipped with eight NVIDIA A100 GPUs. Each code is performed by the corresponding shell script which includes the complete details of hyperparameter and model introduction. The results have been presented using the state-of-art models:

<img width="807" height="465" alt="截屏2025-11-15 22 59 47" src="https://github.com/user-attachments/assets/8c732b09-0a06-47d6-8a68-ca70a2fe56b3" />

<img width="843" height="376" alt="image" src="https://github.com/user-attachments/assets/2d17535c-e9a2-4d63-b380-db9d0be9b4ed" />

Load Google ViT NPZ Weights (https://github.com/xxayt/ViT-for-Cifar100?tab=readme-ov-file)

ViT_B/32: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
ViT_L/32:https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz
ViT_L/16:https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz

## Citation
```bibtex
@misc{Shun Wang  and Shun-Li Shang  and Zi-Kui Liu  and Wenrui Hao,
title = {ZENN: A thermodynamics-inspired computational framework for heterogeneous data–driven modeling},
journal = {Proceedings of the National Academy of Sciences},
volume = {123},
number = {1},
pages = {e2511227122},
year = {2026},
doi = {10.1073/pnas.2511227122},
URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2511227122} 
}


