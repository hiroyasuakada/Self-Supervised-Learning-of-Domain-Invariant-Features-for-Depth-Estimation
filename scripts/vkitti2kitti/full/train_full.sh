set -ex
python train.py \
\
--project_name vkitti2kitti \
--experiment_name vkitti2kitti_full \
--model full \
--num_threads 32 \
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
--niter 10 --niter_decay 10 \
--shuffle --flip --rotation \
--no_html --display_id -1 \
--batch_size 16 \
--gpu_ids 0,1,2,3,4,5,6,7 \
--load_size 960 288 \
\
--lambda_identity 100 \
--lambda_task 100 \
--lambda_crdoco 1.0 \
\
--lr_task 0.0004 \
--lr_trans 0.0002 \
--pre_trained_on_ImageNet \

