set -ex
python train.py \
\
--project_name vkitti2kitti \
--experiment_name vkitti2kitti_cl-dec \
--model cl_dec \
--num_threads 64 \
\
--dataset kitti \
--garg_crop \
--img_source_dir ../datasets/vkitti_data/train_color \
--lab_source_dir ../datasets/vkitti_data/train_depth \
--img_target_dir ./datasplit/eigen_train_files.txt \
--lab_target_dir ./datasplit/eigen_train_files.txt \
--img_target_val_dir ./datasplit/eigen_test_files.txt \
--lab_target_val_dir ./datasplit/eigen_test_files.txt \
--txt_data_path ../datasets/kitti_data/ \
\
--niter 50 --niter_decay 50 \
--shuffle --flip --rotation \
--no_html --display_id -1 \
--batch_size 128 \
--gpu_ids 0,1,2,3,4,5,6,7 \
--load_size 960 288 \
\
--lambda_simsiam 1.0 \
\
--lr_task 0.0032 \
--lr_trans 0.0016 \
--pre_trained_on_ImageNet \
\
--path_to_pre_trained_G_s2t  ./checkpoints/vkitti2kitti_full/20_net_G_s2t.pth \
--path_to_pre_trained_G_t2s  ./checkpoints/vkitti2kitti_full/20_net_G_t2s.pth \
\
--unet_feature_id -1 \
--scale_simsiam_hidden 3 \