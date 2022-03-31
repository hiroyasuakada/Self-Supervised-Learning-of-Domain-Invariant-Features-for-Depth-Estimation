'''
python evaluate.py \
    --gt_path ../datasets/nyu_data/test_depth \
    --split indoor --eigen_crop \
    --min_depth 0.1 --max_depth 10.0 --normalize_depth 10.0 \
    --predicted_depth_path ./results/[] \

python evaluate.py \
    --gt_path ../datasets/kitti_data/ \
    --split eigen --garg_crop \
    --min_depth 1.0 --max_depth 80.0 --normalize_depth 80.0 \
    --predicted_depth_path ./results/[] \

python evaluate.py \
    --gt_path ../datasets/Gridlaserdata/ \
    --split make3d --garg_crop \
    --min_depth 0.001 --max_depth 70.0 --normalize_depth 80.0 \
    --predicted_depth_path ./results/[] \

'''

import argparse
from util.process_data import *
import cv2
import scipy
import scipy.io
from natsort import natsorted

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--split', type=str, default='indoor', help='data split, indoor or eigen')
parser.add_argument('--predicted_depth_path', type=str, 
                    default='results/>>>', help='path to estimated depth')
parser.add_argument('--gt_path', type = str, 
                    default='../datasets/nyu_data/test_depth/', help = 'path to ground truth')
parser.add_argument('--file_path', type = str, default='./datasplit/', help = 'path to datasplit files')
parser.add_argument('--min_depth', type=float, default=0.1, help='minimun depth for evaluation, indoor 0.1 / eigen 1.0 / make3d 0.001')
parser.add_argument('--max_depth', type=float, default=10.0, help='maximun depth for evaluation, indoor 10.0 / eigen 50.0 / make3d 70.0')
parser.add_argument('--normalize_depth', type=float, default=10.0, help='depth normalization value, indoor 10.0 / eigen 80.0 / make3d 80.0')
parser.add_argument('--eigen_crop',action='store_true', help='if set, crops according to Eigen NIPS14')
parser.add_argument('--garg_crop', action='store_true', help='if set, crops according to Garg  ECCV16')
args = parser.parse_args()

if __name__ == "__main__":

    predicted_depths = load_depth(args.predicted_depth_path,args.split, args.normalize_depth)

    if args.split == 'indoor':
        ground_truths = load_depth(args.gt_path, args.split, 10.0)
        num_samples = len(ground_truths)

    elif args.split == 'eigen':
        test_files = natsorted(read_text_lines(args.file_path + 'eigen_test_files.txt'))
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)

        num_samples = len(im_files)
        ground_truths = []

        for t_id in range(num_samples):
            camera_id = cams[t_id]
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            ground_truths.append(depth.astype(np.float32))
        
            depth = cv2.resize(predicted_depths[t_id],(im_sizes[t_id][1], im_sizes[t_id][0]),interpolation=cv2.INTER_LINEAR)            
            predicted_depths[t_id] = depth

        #     # convert dist to depth maps
        #     depth, depth_inter = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, True, True)
        #     ground_truths.append(depth_inter.astype(np.float32))
        #     depth_img = Image.fromarray(np.uint8(depth_inter/80*255))
        #     depth_path = os.path.join('../datasets/kitti_data/eigen_val_labels', str(t_id) + '_' + test_files[t_id].replace('/', '_')[0:66])
        #     depth_img.save(depth_path)

        #     if t_id % 200 == 0:
        #         print(t_id)

        # print('saved')
        # x = input()

    elif args.split == 'make3d':
        with open(os.path.join(args.file_path, "make3d_test_files.txt")) as f:
            test_filenames = f.read().splitlines()
        test_filenames = map(lambda x: x[4:-4], test_filenames)

        ground_truths = []
        for filename in test_filenames:
            mat = scipy.io.loadmat(os.path.join(args.gt_path, "depth_sph_corr-{}.mat".format(filename)))  # "datasets/Gridlaserdata/"
            ground_truths.append(mat["Position3DGrid"][:,:,3])
        
        num_samples = len(ground_truths)
        
        depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), ground_truths)
        ground_truths = list(map(lambda x: x[int((55 - 21)/2): int((55 + 21)/2)], ground_truths))

    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples,np.float32)
    rmse = np.zeros(num_samples,np.float32)
    rmse_log = np.zeros(num_samples,np.float32)
    log_10 = np.zeros(num_samples,np.float32)
    a1 = np.zeros(num_samples,np.float32)
    a2 = np.zeros(num_samples,np.float32)
    a3 = np.zeros(num_samples,np.float32)


    for i in range(num_samples):
    # for i in range(1):
        ground_depth = ground_truths[i]
        predicted_depth = predicted_depths[i]

        if args.split == 'indoor' or args.split == 'eigen':

            height, width = ground_depth.shape
            _height, _width = predicted_depth.shape

            if not height == _height:
                predicted_depth = cv2.resize(predicted_depth,(width,height),interpolation=cv2.INTER_LINEAR)

            mask = np.logical_and(ground_depth > args.min_depth, ground_depth < args.max_depth)

            # crop used by Garg ECCV16
            if args.garg_crop:
                crop = np.array([0.40810811 * height,  0.99189189 * height,
                                        0.03594771 * width,   0.96405229 * width]).astype(np.int32)

            # crop we found by trail and error to reproduce Eigen NIPS14 results
            elif args.eigen_crop:
                crop = np.array([0.3324324 * height,  0.91351351 * height,
                                        0.0359477 * width,   0.96405229 * width]).astype(np.int32)
            

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            ground_depth = ground_depth[mask]
            predicted_depth = predicted_depth[mask]


        elif args.split == 'make3d':
            predicted_depth = cv2.resize(predicted_depth, ground_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(ground_depth > args.min_depth, ground_depth < args.max_depth)
            ground_depth = ground_depth[mask]
            predicted_depth = predicted_depth[mask]
            predicted_depth *= np.median(ground_depth) / np.median(predicted_depth)


        predicted_depth[predicted_depth < args.min_depth] = args.min_depth
        predicted_depth[predicted_depth > args.max_depth] = args.max_depth

        abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], log_10[i], a1[i], a2[i], a3[i] = compute_errors(ground_depth,predicted_depth)

        print('{},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}, {:10.4f}'
            .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], log_10[i], a1[i], a2[i], a3[i]))

    print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','log_10', 'a1','a2','a3'))
    print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
           .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(), log_10.mean(), a1.mean(),a2.mean(),a3.mean()))



