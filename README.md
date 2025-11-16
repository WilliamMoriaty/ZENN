# ZENN
 Zentropy-enhanced neural network

<img width="416" alt="image" src="https://github.com/user-attachments/assets/2d0a4fb6-dfdc-4c33-8b7c-9cbd514b94e1" />

Schematic of ZENN and its applications in different areas. Zentropy theory integrates statistical mechanics and quantum mechanics by assigning intrinsic entropy to each system component, thereby capturing internal disparities. By embedding zentropy theory into deep learning as a backward modeling framework, ZENN replaces the internal energy E^((k)) and S^((k)) of each configuration with simple neural networks, and integrates information across all configurations through the total free energy F. In this paper, ZENN has been applied in three representative tasks—multi-source data integration, energy landscape reconstruction, and inference of Fe₃Pt alloy properties—demonstrating its potential as a powerful framework that effectively bridges statistical mechanics and machine learning.

Table 1 Test accuracy in CIFAR-10 
	ZENN	CE
	N_T=3	N_T=4	N_T=5	N_T=6	
ViT-B/32	99.16%	99.14%	99.34%	99.00%	98.79%
ViT-L/32	99.40%	99.47%	99.47%	99.35%	99.24%
ViT-L/16	99.47%	99.48%	99.51%	99.48%	99.36%

Table 2 Test accuracy in CIFAR-100 
	ZENN	CE
	N_T=2	N_T=3	N_T=4	N_T=5	
ViT-B/32	93.52%	93.14%	93.10%	93.31%	92.03%
ViT-L/32	94.68%	94.08%	94.10%	93.99%	93.17%
ViT-L/16	95.75%	95.69%	95.37%	95.14%	93.86%

Table 3 Test accuracy in BBCNews 
	ZENN	CE
	N_T=2	N_T=3	N_T=4	N_T=5	
SmolLM2-135M	99.18%	98.80%	98.75%	98.68%	97.20%
SmolLM2-360M	99.40%	98.73%	98.72%	98.70%	98.20%

Table 1 Test accuracy in AGNews 
	ZENN	CE
	N_T=2	N_T=3	N_T=4	N_T=5	
SmolLM2-135M	98.06%	97.87%	98.45%	98.16%	94.33%
SmolLM2-360M	98.28%	98.26%	98.50%	98.30%	94.83%

<img width="415" height="432" alt="image" src="https://github.com/user-attachments/assets/592b8d66-eab3-411a-9e1f-3531f8a19deb" />
