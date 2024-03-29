# Self-Supervised Learning of Domain Invariant Features for Depth Estimation (WACV2022)

Official PyTorch implementation of our WACV 2022 paper, "Self-Supervised Learning of Domain Invariant Features for Depth Estimation".

> **Self-Supervised Learning of Domain Invariant Features for Depth Estimation**<br>
Hiroyasu Akada, Shariq Farooq Bhat, Ibraheem Alhashim, Peter Wonka<br>
> [Paper link](https://openaccess.thecvf.com/content/WACV2022/html/Akada_Self-Supervised_Learning_of_Domain_Invariant_Features_for_Depth_Estimation_WACV_2022_paper.html)<br><br>

**For any questions, please contact the first author, Hiroyasu Akada [hakada@mpi-inf.mpg.de] .**

## Citation: 

    @InProceedings{Akada_2022_WACV,
    author    = {Akada, Hiroyasu and Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
    title     = {Self-Supervised Learning of Domain Invariant Features for Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3377-3387}
    }

## Requirements 

- Linux environment
- 8 NVIDIA A100 GPUs
- Docker: please use the provided [dockerfile](https://github.com/hiroyasuakada/Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022/blob/main/docker_AWS_a100/dockerfile) to install dependencies.
- Download this repository

        git clone https://github.com/hiroyasuakada/Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022.git
        
        cd Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022


## Datasets
The outdoor synthetic dataset is vKITTI and outdoor realistic dataset is KITTI. We use KITTI and Make3D datasets for evaluation. 

For dataset pre-processing, we follow the same way as previous works. Therefore, please follow the instructions by [Monodepth](https://github.com/mrharicot/monodepth) and [T2Net](https://github.com/lyndonzheng/Synthetic2Realistic).  

After the dataset pre-processing, please place the datasets in this repository like below.

| Folder relation ;
| :--- 
| Self-Supervised-Learning-of-Domain-Invariant-Features-for-Depth-Estimation-WACV2022
| &boxur;&nbsp; datasets
| &ensp;&ensp; &boxur;&nbsp; kitti_data
| &ensp;&ensp; &boxur;&nbsp; vkitti_data
| &ensp;&ensp; &boxur;&nbsp; make3d


## Training
### Style transfer stage

        bash scripts/vkitti2kitti/full/train_full.sh
        
        
###  SSRL stage 

        bash scripts/vkitti2kitti/cl_dec/train_cl-dec.sh


###  Fine-tuning stage 

        bash scripts/vkitti2kitti/full_fine/train_full_fine.sh
 
### Trained weights

You can download the [trained weight](https://drive.google.com/file/d/1iB1KuwpysEND2r1iqcpppz7aTBbX_F5E/view?usp=sharing) of our task network. Please place the weight in `./checkpoints/vkitti2kitti_full-fine/`.

## Testing (generate images)
### KITTI

        bash scripts/vkitti2kitti/test/test_kitti.sh

### Make3D

        bash scripts/vkitti2kitti/test/test_make3d.sh


## Evaluation
Please look at evaluate.py.

You need to specify the foloder that contains predicted images in '--predicted_depth_path'.



## Acknowledgments
Code is inspired by [T2Net (Synthetic2Realistic)](https://github.com/lyndonzheng/Synthetic2Realistic.git).



