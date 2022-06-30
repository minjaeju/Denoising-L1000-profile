# A denoised embedding space of genetic perturbation using Deep Metric Learning
This repository provides the official PyTorch implementation for the following paper:

**A denoised embedding space of genetic perturbation using Deep Metric Learning**<br>
Minjae Ju; Sanghoon Lee; Jaewoo Kang<br>
in 2022 IEEE International Conference on Big Data and Smart Computing (BigComp)<br>
[**Paper**](https://ieeexplore.ieee.org/document/9736519) 
>**Abstract:** Identifying and relieving internal noises of expression profile is crucial in drug discovery. Among various perturbagens, the most common cause of off-target effects in genetic perturbation is known as seed effects. In this paper, we propose a model to denoise seed effects in LINCS/L1000 gene knock down (KD) dataset by using deep metric learning. Results show that our model can embed profiles with the identical gene target into similar embedding spaces, whereas profiles with the same seed sequence but with different gene targets can embed farther away. This robust embedding space could help reveal the mechanism of actions (MoA) of compounds or solve other downstream tasks using expression profiles.*

Requirements
------------
Install [PyTorch](https://pytorch.org/)

Reference
---------

If you find the code useful, please cite our papers.

    @inproceedings{ju2022denoised,
            title={A denoised embedding space of genetic perturbation using Deep Metric Learning},
            author={Ju, Minjae and Lee, Sanghoon and Kang, Jaewoo},
            booktitle={2022 IEEE International Conference on Big Data and Smart Computing (BigComp)},
            pages={378--381},
            year={2022},
            organization={IEEE}
        }
