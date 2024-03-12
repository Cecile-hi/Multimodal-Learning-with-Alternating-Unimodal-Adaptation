# Multimodal Learning with Alternating Unimodal Adaptation

This is the official project of the Multimodal Learning with Alternating Unimodal Adaptation (MLA) method proposed by our paper titled 'Multimodal representation learning by alternating unimodal adaptation' published on the 40th The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR 2024) (https://arxiv.org/pdf/2311.10707.pdf).

## Citation

If you find this toolkit useful, please consider citing following papers.
```
@article{zhang2023multimodal,
  title={Multimodal representation learning by alternating unimodal adaptation},
  author={Zhang, Xiaohui and Yoon, Jaehong and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2311.10707},
  year={2023}
}

```
### Introduction 
Existing multimodal learning methods often struggle with challenges where some modalities appear more
dominant than others during multimodal learning, resulting in suboptimal performance. To address this challenge, we propose MLA (Multimodal Learning with Alternating Unimodal Adaptation). MLA reframes the conventional joint multimodal learning process by transforming it into an alternating unimodal learning process, thereby minimizing interference between modalities. Simultaneously, it captures cross-modal interactions through a shared head, which undergoes continuous optimization across different modalities.
This optimization process is controlled by a gradient modification mechanism to prevent the shared head from losing previously acquired information. During the inference phase, MLA utilizes a test-time uncertainty-based model fusion mechanism to integrate multimodal information.

![](./images/MLA_framework.PNG)

<!-- <img src="./subspaces.jpg" width="50%"> -->

### For training on CREMA-D with audio-video modalities:  
#### Normal
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb base --modulation Normal --epochs 100 --dataset CREMAD
```
#### OGM (-GE)
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb base --modulation OGM (-GE) --epochs 100 --dataset CREMAD
```
#### QMF
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb base --modulation QMF --epochs 100 --dataset CREMAD
```
#### MLA (fixed fusion)
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb base --modulation Normal --epochs 100 --dataset CREMAD --gs_flag
```
#### MLA (dynamic fusion)
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb base --modulation Normal --epochs 100 --dataset CREMAD --gs_flag -dynamic
```
### For training on Food-101 (MVSA) with video-text modalities:
Sames as the command of CREMA-D, with a few change.
For example:
#### MLA (dynamic fusion)
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb m3ae --modulation Normal --epochs 100 --dataset Food101 (MVSA) --gs_flag -dynamic
```
### For training on IEMOCAP with audio-video-text modalities:
Sames as the command of CREMA-D, with a few change.
For example:
#### MLA (dynamic fusion)
```
  python main.py --train --ckpt_path ckpt --gpu_ids 0 --batch_size 64 --lorb m3ae --modulation Normal --epochs 100 --dataset IEMOCAP --gs_flag -dynamic --modal3
```
### Training using CLIP feature
Comming soon...
