# Self-Supervised Learning of Domain Invariant Features for Depth Estimation (WACV2022)

A PyTorch implementation of our WACV 2022 paper, "Self-Supervised Learning of Domain Invariant Features for Depth Estimation".

For any questions, please contact the first author, Hiroyasu Akada [hakada@mpi-inf.mpg.de] .


> **Self-Supervised Learning of Domain Invariant Features for Depth Estimation**<br>
Hiroyasu Akada, Shariq Farooq Bhat, Ibraheem Alhashim, Peter Wonka<br>
> Paper: https://openaccess.thecvf.com/content/WACV2022/html/Akada_Self-Supervised_Learning_of_Domain_Invariant_Features_for_Depth_Estimation_WACV_2022_paper.html<br><br>
> Abstract: *We tackle the problem of unsupervised synthetic-to-real domain adaptation for single image depth estimation. An essential building block of single image depth estimation is an encoder-decoder task network that takes RGB images as input and produces depth maps as output. In this paper, we propose a novel training strategy to force the task network to learn domain invariant representations in a self-supervised manner. Specifically, we extend self-supervised learning from traditional representation learning, which works on images from a single domain, to domain invariant representation learning, which works on images from two different domains by utilizing an image-to-image translation network. Firstly, we use an image-to-image translation network to transfer domain-specific styles between synthetic and real domains. This style transfer operation allows us to obtain similar images from the different domains. Secondly, we jointly train our task network and Siamese network with the same images from the different domains to obtain domain invariance for the task network. Finally, we fine-tune the task network using labeled synthetic and unlabeled real-world data. Our training strategy yields improved generalization capability in the real-world domain. We carry out an extensive evaluation on two popular datasets for depth estimation, KITTI and Make3D. The results demonstrate that our proposed method outperforms the state-of-the-art on all metrics, e.g. by 14.7% on Sq Rel on KITTI. The source code and model weights will be made available.*

# Requirements 

- Linux environment
- 8 NVIDIA A100 GPUs
- Docker: [provided Dockerfile](https://github.com/hiroyasuakada/Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022/blob/main/docker_AWS_a100/dockerfile)
- Download this repository

        git clone https://github.com/hiroyasuakada/Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022.git
        
        cd Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022


# Datasets
The outdooe Synthetic Dataset is vKITTI and outdoor Realistic dataset is KITTI. We use KITTI and Make3D datasets for evaluation. 

For dataset preparation, please look at previous works, [Monodepth](https://github.com/mrharicot/monodepth) and [T2Net](https://github.com/lyndonzheng/Synthetic2Realistic).  


# Training
## Style transfer stage

        bash scripts/vkitti2kitti/full/train_full.sh
        
        
##  SSRL stage 

        bash scripts/vkitti2kitti/cl_dec/train_cl-dec.sh


##  fine-tuning stage 

        bash scripts/vkitti2kitti/full_fine/train_full_fine.sh
 
## Trained weights

The trained weights will be available soon.




