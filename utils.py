import os
import shutil
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf


def collect_npy_data(folder, out_folder, fimg, fdepth, window, stride, channel):
    elements = fimg.split('_')
    dt = os.path.basename(elements[2])

    # open file
    with rasterio.open(folder+fimg) as img:
        arr_img = img.read()
    
    with rasterio.open(folder+fdepth) as depth:
        arr_depth = depth.read()

    # img tiling
    img_stack = moving_window(arr_img, window_size=(window, window), steps=(1, 1), channel=channel)
    np.save(out_folder+'rgbnss_'+dt, img_stack, allow_pickle=True)
    img_stack = moving_window(arr_img, window_size=(window, window), steps=(stride, stride), channel=channel)

    # depth tiling
    depth_stack = moving_window(arr_depth, window_size=(window, window), steps=(1, 1), channel=1)
    depth_stack = np.float32(depth_stack)
    np.save(out_folder+'depth_'+dt, depth_stack, allow_pickle=True)
    depth_stack = moving_window(arr_depth, window_size=(window, window), steps=(stride, stride), channel=1)
    
    # remove nodata values
    img_nodata = np.float32(img.nodata)
    depth_nodata = np.float32(depth.nodata)
    base = window*window*channel    # 5x5 with RGB have 75 elements inside
    # count array w/o nodata -> should be equal with base number
    unique = np.array(np.unique(np.argwhere(img_stack!=img_nodata)[:,0], return_counts=True)).T
    img_idx = np.argwhere(unique==base)[:,0]    # extract only the indexes
    img_stack = img_stack[img_idx,:,:,:]
    depth_stack = depth_stack[img_idx,:]
    depth_idx = np.nonzero(depth_stack!=depth_nodata)[0]
    img_stack = img_stack[depth_idx,:,:,:]
    depth_stack = depth_stack[depth_idx,:]
    # remove zero depths
    depth_idx = np.nonzero(depth_stack!=0.0)[0]
    img_stack = img_stack[depth_idx,:,:,:]
    depth_stack = depth_stack[depth_idx,:]
    # remove positive depths (>2m)
    depth_idx = np.nonzero(depth_stack<2.0)[0]
    img_stack = img_stack[depth_idx,:,:,:]
    depth_stack = depth_stack[depth_idx,:]

    # split into training and testing sets
    n = depth_stack.shape[0]
    tst_idx = np.random.choice(range(n), size=int(0.2*n), replace=False)
    # m = list(set(range(n)))
    trn_idx = list(set(range(n)) - set(tst_idx))
    img_trn, img_tst = img_stack[trn_idx,:,:,:], img_stack[tst_idx,:,:,:]
    depth_trn, depth_tst = depth_stack[trn_idx,:], depth_stack[tst_idx,:]

    # save as npy
    np.save(out_folder+'rgbnss_trn_'+dt, img_trn, allow_pickle=True)
    np.save(out_folder+'rgbnss_tst_'+dt, img_tst, allow_pickle=True)
    np.save(out_folder+'depth_trn_'+dt, depth_trn, allow_pickle=True)
    np.save(out_folder+'depth_tst_'+dt, depth_tst, allow_pickle=True)


def moving_window(arr, window_size, steps, channel):
    """Adapted from https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
    """
    in_shape = np.array(arr.shape[-len(steps):])
    window_size = np.array(window_size)
    steps = np.array(steps)
    nbytes = arr.strides[-1]

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_size) // steps + 1)
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_size)

    tiles = np.lib.stride_tricks.as_strided(arr, shape=outshape, strides=strides, writeable=False)
    stack = tiles.reshape(tiles.shape[0]*tiles.shape[1], window_size[0], window_size[1], channel)

    if channel == 1:
        stack = stack[:, 2, 2, :]
    
    return stack


def merged_tiles(arr, row, col, window_size):
    r_extra = np.floor(window_size[0] / 2).astype(int)
    c_extra = np.floor(window_size[1] / 2).astype(int)

    arr_in = np.reshape(arr, (row - 2*r_extra, col - 2*c_extra, 1))

    arr_out = np.zeros((row, col, 1), dtype=arr_in.dtype)
    arr_out[r_extra:-r_extra, c_extra:-c_extra, :] = arr_in
    arr_out = np.reshape(arr_out, (row, col))
    return arr_out


def write_tif(pathin, pathout, fin, fout, arr):
    with rasterio.open(pathin+fin) as src:
        profile = src.profile
    
    with rasterio.open(pathout+fout+'.tif', 'w', **profile) as dst:
        dst.write(arr, indexes=1)


def sdb_cnn(input_size = (9, 9, 6), dropout_rate=0.3):
  i = tf.keras.Input(input_size)
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid')(i)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='valid')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='valid')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(dropout_rate)(x)
  x = tf.keras.layers.Dense(1, activation='linear')(x)

  model = tf.keras.Model(inputs=i, outputs=x)
  return model
