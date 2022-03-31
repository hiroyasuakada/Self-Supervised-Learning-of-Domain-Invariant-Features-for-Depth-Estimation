import numpy as np
import matplotlib.pyplot as plt
import OpenEXR, Imath, array
import PIL
from PIL import Image
import os, glob
import cv2
from pathlib import Path
from natsort import natsorted
import math, random

def exr2array(exr_file, gray=False):
    # load files
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_exr = OpenEXR.InputFile(exr_file)
    # print(img_exr.header())

    # extract rgb values
    r_str, g_str, b_str = img_exr.channels('RGB', pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))

    # get image size
    dw = img_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # align with opencv format 
    img = np.array([[r, g, b] for r, g, b in zip(red, green, blue)], dtype=np.float)
    img = img.reshape(size[1], size[0], 3)

    if gray:  # 0 - 1000 to 0.0 - 1.0
        img = np.clip(img / 1000, 0.0, 1.0)
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]  # rgb to gray
    else:  # 0.0 - 1.0
        img = np.clip(img, 0.0, 1.0)

    # if gray:  # 0 - 1000 to 0 - 255
    #     img = np.clip(img / 1000, 0, 1) * 255
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # rgb to gray
    # else:  # 0 - 1 to 0 - 255
    #     img = np.clip(img, 0, 1) * 255

    return img

    # print(type(img))
    # print(img.shape)  # (1024, 1366, 3)
    # print(img.dtype)
    # print(img)
    # print(img[:, :, 0])
    # print('max: {}'.format(np.max(img)))
    # print('min: {}'.format(np.min(img)))
    # # 画像を表示
    # plt.figure(dpi=300)
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    dir = './unreal_data_kaust/data'
    folders = os.listdir(dir)
    save_dir = './unreal_data_png'
    save_folders = ['train_color', 'train_depth', 'test_color', 'test_depth']
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, save_folders[0])):
        os.mkdir(os.path.join(save_dir, save_folders[0]))
    if not os.path.exists(os.path.join(save_dir, save_folders[1])):
        os.mkdir(os.path.join(save_dir, save_folders[1]))
    if not os.path.exists(os.path.join(save_dir, save_folders[2])):
        os.mkdir(os.path.join(save_dir, save_folders[2]))
    if not os.path.exists(os.path.join(save_dir, save_folders[3])):
        os.mkdir(os.path.join(save_dir, save_folders[3]))

    for folder in folders:
        path_exr_color = natsorted(glob.glob(os.path.join(dir, folder, '*FinalImage.exr')))
        path_exr_depth = natsorted(glob.glob(os.path.join(dir, folder, '*SceneDepth.exr')))
        assert len(path_exr_color) == len(path_exr_depth)

        # prepare test dataset, 1% of total images
        len_test = math.ceil(len(path_exr_color) / 100)
        list_nodup = []
        while len(list_nodup) < len_test:
            n = random.randint(0, len(path_exr_color) - 1)
            if not n in list_nodup:
                list_nodup.append(n)

        # exr to rgb
        for id in range(len(path_exr_color)):       
            exr_color = path_exr_color[id]
            exr_depth = path_exr_depth[id]

            basename_color = os.path.basename(exr_color)
            basename_depth = os.path.basename(exr_depth)
            print(basename_color)
            print(basename_depth)
            
            jpg_color = exr2jpg(exr_color)
            jpg_depth = exr2jpg(exr_depth, gray=True)

            if id in list_nodup:  # save rgb and depth as test dataset
                cv2.imwrite(os.path.join(save_dir, save_folders[2], basename_color.replace('.exr', '.png')), jpg_color)
                cv2.imwrite(os.path.join(save_dir, save_folders[3], basename_depth.replace('.exr', '.png')), jpg_depth)
            else:  # save rgb and depth as train dataset
                cv2.imwrite(os.path.join(save_dir, save_folders[0], basename_color.replace('.exr', '.png')), jpg_color)
                cv2.imwrite(os.path.join(save_dir, save_folders[1], basename_depth.replace('.exr', '.png')), jpg_depth)

        # with open(os.path.join(path_to_seg, 'train_syn_with_label.txt'), mode='a') as f:
        #     for file in files:
        #         get_file_name = os.path.basename(file)
        #         f.write('{}\n'.format('gta5/img/' + get_file_name))
        #     print('finished!')


###################################################################################

# ### if depth:
# img = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]


# def exr2rgb(exrfile, jpgfile):
#     # load exr data
#     img_exr = OpenEXR.InputFile(exrfile)
#     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#     print(img_exr.header())

#     # get image size
#     dw = img_exr.header()['dataWindow']
#     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

#     # get rgb
#     rgbf = [Image.frombytes("F", size, img_exr.channel(c, pt)) for c in "RGB"]

#     # normalize
#     extrema = [im.getextrema() for im in rgbf]
#     darkest = min([lo for (lo,hi) in extrema])
#     lighest = max([hi for (lo,hi) in extrema])
#     scale = 255 / (lighest - darkest)
#     def normalize_0_255(v):
#         return (v * scale) + darkest
#     rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
#     img_jpg = Image.merge("RGB", rgb8)

#     # show details of the converted jpg image
#     print(type(img_jpg))
#     print(img_jpg.size)
#     print(img_jpg.mode)
#     print(img_jpg)
#     img_jpg.show()

#     # save jpg image
#     img_jpg.save(jpgfile)

# def exr2gray(exrfile, jpgfile):
#     # load exr data
#     img_exr = OpenEXR.InputFile(exrfile)
#     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#     print(img_exr.header())

#     # get image size
#     dw = img_exr.header()['dataWindow']
#     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

#     # get rgb
#     rgbf = [Image.frombytes("F", size, img_exr.channel(c, pt)) for c in "RGB"]

#     # normalize
#     extrema = [im.getextrema() for im in rgbf]
#     darkest = min([lo for (lo,hi) in extrema])
#     lighest = max([hi for (lo,hi) in extrema])
#     scale = 255 / (lighest - darkest)
#     def normalize_0_255(v):
#         return (v * scale) + darkest
#     rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
#     img_jpg = Image.merge("RGB", rgb8)

#     # convert image to grayscale
#     img_jpg = img_jpg.convert('L')

#     # show details of the converted jpg image
#     print(type(img_jpg))
#     print(img_jpg.size)
#     print(img_jpg.mode)
#     print(img_jpg)
#     img_jpg.show()


#     # save jpg image
#     img_jpg.save(jpgfile)

# if __name__ == "__main__":
#     exr2rgb(img_path_color, 'test_unreal_color.jpg') 
#     print('=============================================================')
#     exr2gray(img_path_gray, 'test_unreal_gray.jpg') 
