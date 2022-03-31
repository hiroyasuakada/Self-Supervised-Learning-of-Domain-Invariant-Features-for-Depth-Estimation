import os
import os.path
from natsort import natsorted

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.exr'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(opt, path_files, data_list=None):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(opt, path_files, data_list)
    else:
        paths, size = make_dataset_dir(opt, path_files, data_list)

    return paths, size

def make_dataset_txt(opt=None, path_files=None, data_list=None):
    # reading txt file
    image_paths = []

    with open(path_files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        path_left = os.path.join(opt.txt_data_path, path[0:66].replace('.jpg', '.png'))
        # path_right = os.path.join(opt.txt_data_path, path[67:].replace('.jpg', '.png'))
        image_paths.append(path_left)
        # image_paths.append(path_right)

    if opt is not None:
        if opt.experiment:
            image_paths = image_paths[:165]
    
    # For training with less datasets designated in the data_list
    if data_list:
        with open(data_list) as f:
            ids = f.readlines()
        
        image_paths_new = []
        for id in ids:
            image_paths_new.append(image_paths[int(id)])
        image_paths = image_paths_new

    return image_paths, len(image_paths)

def make_dataset_dir(opt=None, dir=None, data_list=None):
    image_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for fname in natsorted(os.listdir(dir)):
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            image_paths.append(path)

    if opt is not None:
        if opt.experiment:
            image_paths = image_paths[:165]
    
    # For training with less datasets designated in the data_list
    if data_list:
        with open(data_list) as f:
            ids = f.readlines()
        
        image_paths_new = []
        for id in ids:
            image_paths_new.append(image_paths[int(id)])
        image_paths = image_paths_new

    return image_paths, len(image_paths)













# import os
# import os.path

# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# def make_dataset(path_files):
#     if path_files.find('.txt') != -1:
#         paths, size = make_dataset_txt(path_files)
#     else:
#         paths, size = make_dataset_dir(path_files)

#     return paths, size

# def make_dataset_txt(path_files):
#     # reading txt file
#     image_paths = []

#     with open(path_files) as f:
#         paths = f.readlines()

#     for path in paths:
#         path = path.strip()
#         image_paths.append(path)

#     return image_paths, len(image_paths)


# def make_dataset_dir(dir):
#     image_paths = []

#     assert os.path.isdir(dir), '%s is not a valid directory' % dir

#     for root, _, fnames in os.walk(dir):
#         for fname in sorted(fnames):
#             if is_image_file(fname):
#                 path = os.path.join(root, fname)
#                 image_paths.append(path)

#     return image_paths, len(image_paths)