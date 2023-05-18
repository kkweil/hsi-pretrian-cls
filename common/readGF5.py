import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
import shutil

def read_gf5(path_vn, path_sw, remove_ruins=True, scale=None, n_components=None):
    # sw: 1-150 vn 151-330
    # ruins bands: 193-203,246-262,326-330

    dataset_sw = gdal.Open(path_sw)
    im_width = dataset_sw.RasterXSize
    im_height = dataset_sw.RasterYSize
    im_data_sw = dataset_sw.ReadAsArray(0, 0, im_width, im_height)

    dataset_vn = gdal.Open(path_vn)
    im_width = dataset_vn.RasterXSize
    im_height = dataset_vn.RasterYSize
    im_data_vn = dataset_vn.ReadAsArray(0, 0, im_width, im_height)
    im_data = np.concatenate((im_data_vn, im_data_sw), axis=0)
    if remove_ruins:
        ruins_bands = list(range(193 - 1, 203)) + list(range(246 - 1, 262)) + list(range(326 - 1, 330))
        im_data = np.delete(im_data, ruins_bands, axis=0)
    if scale:
        shape = im_data.shape  # c,w,h
        im_data = im_data.reshape(im_data.shape[0], -1).transpose()
        im_data = scale(im_data).transpose().reshape(-1, shape[1], shape[2])  # .astype('float32')
    if n_components:
        shape = im_data.shape
        im_data = im_data.reshape(shape[0], -1).transpose()
        pca = PCA(n_components)
        im_data = pca.fit_transform(im_data)
        print(pca.explained_variance_ratio_)
    return im_data


def split(im_data, patch_size=None, save_path=None):
    horizontal_steps, vertical_steps = im_data.shape[1] // patch_size, im_data.shape[2] // patch_size  # 防止越界
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('Save data in ' + save_path)
    nums = 0
    print('Start splitting...')
    for i in range(horizontal_steps):
        for j in range(vertical_steps):
            nums += 1
            np.save(save_path + '/' + save_path.split('/')[-1] + f'{patch_size}_' + f'{nums}'.zfill(6),
                    im_data[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size].astype(
                        'float32'))
    print(f'Done! Total {patch_size}_patches {nums}')


def readfiles(img_root):
    filelist = []
    for rootdir, subdirs, files in os.walk(img_root):
        if files and subdirs == []:
            for i in files:
                filelist.append(os.path.join(rootdir, i))
    return filelist


if __name__ == '__main__':
    # path_sw = r'../data/ShangHai/GF5_AHSI_E121.45_N31.31_20191228_008714_L10000071208_SW.geotiff'
    # path_vn = r'../data/ShangHai/GF5_AHSI_E121.45_N31.31_20191228_008714_L10000071208_VN.geotiff'
    path = {
        'village':
            {'path_vn': r'../data/village/GF5_AHSI_E109.97_N34.77_20190404_004815_L10000039871_SW.geotiff',
             'path_sw': r'../data/village/GF5_AHSI_E109.97_N34.77_20190404_004815_L10000039871_VN.geotiff'},
        'mountain_village':
            {'path_vn': r'../data/mountain_village/GF5_AHSI_E113.12_N39.70_20190703_006123_L10000049630_SW.geotiff',
             'path_sw': r'../data/mountain_village/GF5_AHSI_E113.12_N39.70_20190703_006123_L10000049630_VN.geotiff'},
        'forest':
            {'path_vn': r'../data/forest/GF5_AHSI_E113.19_N37.24_20190906_007070_L10000055322_SW.geotiff',
             'path_sw': r'../data/forest/GF5_AHSI_E113.19_N37.24_20190906_007070_L10000055322_VN.geotiff'},
        'desert':
            {'path_vn': r'../data/desert/GF5_AHSI_E114.20_N40.19_20190422_005078_L10000041710_SW.geotiff',
             'path_sw': r'../data/desert/GF5_AHSI_E114.20_N40.19_20190422_005078_L10000041710_VN.geotiff'},
        'forest_village':
            {'path_vn': r'../data/forest_village/GF5_AHSI_E114.31_N38.04_20190816_006764_L10000053696_SW.geotiff',
             'path_sw': r'../data/forest_village/GF5_AHSI_E114.31_N38.04_20190816_006764_L10000053696_VN.geotiff'},
        'township':
            {'path_vn': r'../data/township/GF5_AHSI_E121.32_N31.81_20191228_008714_L10000069108_SW.geotiff',
             'path_sw': r'../data/township/GF5_AHSI_E121.32_N31.81_20191228_008714_L10000069108_VN.geotiff'},
        'snowfield':
            {'path_vn': r'../data/snowfield/GF5_AHSI_E127.62_N47.06_20191220_008597_L10000068300_SW.geotiff',
             'path_sw': r'../data/snowfield/GF5_AHSI_E127.62_N47.06_20191220_008597_L10000068300_VN.geotiff'},
        'metropolis':
            {'path_vn': r'../data/metropolis/GF5_AHSI_E121.45_N31.31_20191228_008714_L10000071208_SW.geotiff',
             'path_sw': r'../data/metropolis/GF5_AHSI_E121.45_N31.31_20191228_008714_L10000071208_VN.geotiff'},
    }
    # land_types = ['village', 'mountain_village', 'forest', 'desert', 'township', 'forest_village', 'snowfield', 'metropolis']
    patch_size = [9, 15, 23, 31]
    land_types = ['snowfield', 'metropolis']
    # patch_size = [7]
    root = r'../data/GF5_patches/'

    for i in land_types:
        if os.path.exists(r'../data/'+i):
            im_data = read_gf5(path[i]['path_vn'], path[i]['path_sw'], scale=preprocessing.scale)
            for j in patch_size:
                split(im_data, patch_size=j, save_path=root+f'patch{j}/'+i)

    train_root = root + 'train'
    test_root = root + 'test'
    split_rate = 0.9
    for i in patch_size:
        samples = readfiles(root+f'patch{i}')
        random.shuffle(samples)
        tr_size = int(split_rate*len(samples))
        tr = samples[:tr_size]
        te = samples[tr_size:]

        for p in tr:
            dst_p = os.path.join(train_root, p.split('/')[-3], p.split('/')[-2])
            if not os.path.exists(dst_p):
                os.makedirs(dst_p, exist_ok=True)
            shutil.move(p, dst_p)

        for p in te:
            dst_p = os.path.join(test_root, p.split('/')[-3], p.split('/')[-2])
            if not os.path.exists(dst_p):
                os.makedirs(dst_p, exist_ok=True)
            shutil.move(p, dst_p)


    # img = np.load('../data/GF5Patches/ShangHai/000001.npy')
    # plt.imshow(img[50])
    # plt.show()
    a = 0
