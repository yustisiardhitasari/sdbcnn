import os
import sys
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', '-m', default='sdb_cnn_dropout=0.3_lr=0.0001_bsize=512_0300.ckpt', help='Model file name')
    parser.add_argument('--xtst', '-x', default='rgbnss_190108.npy', help='Full sub-images')
    parser.add_argument('--ytst', '-y', default='depth_190108.npy', help='Full sub-images')
    parser.add_argument('--fdepth', '-fd', default='ponce_depth_aoi1_10m_crct.tif', help='Depth reference file')
    parser.add_argument('--window', '-w', default=9, help='Window size of sub images', type=int)

    args = parser.parse_args()
    print(args)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    elements = args.xtst.split('_')
    dt = os.path.basename(elements[-1]).split('.')[0]
    
    # load data
    folder_data = os.path.join(BASE_DIR, 'data/')
    rgb_tst = np.load(folder_data+args.xtst)
    depth_tst = np.load(folder_data+args.ytst)
    print('testing images shape: {}'.format(rgb_tst.shape))
    print('testing depth shape: {}'.format(depth_tst.shape))

    # load model
    model = utils.sdb_cnn(input_size=(9, 9, 6), dropout_rate=0.3)
    folder_ckpt = os.path.join(BASE_DIR, 'ckpts/')
    model.load_weights(folder_ckpt+args.load_model)

    # prediction full image
    depth_pred = model.predict(rgb_tst, verbose=1)
    print('predict depth shape: {}'.format(depth_pred.shape))
    print('test depth shape: {}'.format(depth_tst.shape))

    # open depth reference image
    with rasterio.open(folder_data+args.fdepth) as depth:
        arr_depth = depth.read()
        row = arr_depth.shape[1]
        col = arr_depth.shape[2]
    
    # merged tiles and save as tif
    result = utils.merged_tiles(depth_pred, row, col, window_size=(args.window,args.window))
    utils.write_tif(folder_data, folder_data, args.fdepth, 'sdbcnn_'+dt, result)


if __name__ == "__main__":
    main()