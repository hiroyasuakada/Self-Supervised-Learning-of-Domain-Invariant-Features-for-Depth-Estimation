'''
python evaluate.py \
--predicted_depth_path ../sftp/results/crdoco_v2_end2end_cyclegan_unreal2nyu_crdoco1_gp/test_7/images/img_t_task \
--gt_path ../datasets/nyu_data/test_depth \
--split indoor --eigen_crop

python evaluate.py \
--predicted_depth_path results/crdoco_v2_end2end_vkitti2kitti_crdoco1 \
--gt_path ../datasets/kitti_data \
--split eigen --garg_crop --min_depth 1.0 --max_depth 50.0 --normalize_depth 80.0

'''

import argparse
from util.process_data import *
import cv2
import scipy
from natsort import natsorted

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--split', type=str, default='indoor', help='data split, indoor or eigen')
parser.add_argument('--predicted_depth_path', type=str, 
                    default='results/>>>', help='path to estimated depth')
parser.add_argument('--gt_path', type = str, 
                    default='../datasets/nyu_data/test_depth/', help = 'path to ground truth')
parser.add_argument('--file_path', type = str, default='./datasplit/', help = 'path to datasplit files')
parser.add_argument('--min_depth', type=float, default=0.1, help='minimun depth for evaluation, indoor 0.1 / outdoor 1.0')
parser.add_argument('--max_depth', type=float, default=10.0, help='maximun depth for evaluation, indoor 10.0 / outdoor 50.0')
parser.add_argument('--normalize_depth', type=float, default=10.0, help='depth normalization value, indoor 10.0 / outdoor 80.0')
parser.add_argument('--eigen_crop',action='store_true', help='if set, crops according to Eigen NIPS14')
parser.add_argument('--garg_crop', action='store_true', help='if set, crops according to Garg  ECCV16')
args = parser.parse_args()


if __name__ == "__main__":

    main_path = '../datasets/'

    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()
    test_filenames = map(lambda x: x[4:-4], test_filenames)


    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = 1704 / 2
    depth_new_height = 21
    
    depths_gt = []
    for filename in test_filenames:
        mat = scipy.io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
        depths_gt.append(mat["Position3DGrid"][:,:,3])

    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = map(lambda x: x[(55 - 21)/2:(55 + 21)/2], depths_gt)

    pred_disps = np.load(path_to_pred_disps)

    errors = []
    for i in range(len(test_filenames)):
        depth_gt = depths_gt_cropped[i]
        depth_pred = 1 / pred_disps[i]
        depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        mask = np.logical_and(depth_gt > 0, depth_gt < 70)
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= np.median(depth_gt) / np.median(depth_pred)
        depth_pred[depth_pred > 70] = 70
        errors.append(compute_errors(depth_gt, depth_pred))
    mean_errors = np.mean(errors, 0)

    print(("{:>8} | " * 4).format( "abs_rel", "sq_rel", "rmse", "rmse_log"))
    print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))







