import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import utils
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xtrn', '-x', default='rgbnss_trn_190108.npy', help='Training data')
    parser.add_argument('--ytrn', '-y', default='depth_trn_190108.npy', help='Depth file name')
    parser.add_argument('--epochs', '-e', default=300, help='Number of training epochs', type=int)
    parser.add_argument('--bsize', '-b', default=512, help='Batch size', type=int)
    parser.add_argument('--lr', '-lr', default=0.0001, help='Learning rate', type=int)
    parser.add_argument('--log', help='Log to file in save folder; use - for stdout (default is log.txt)', metavar='file', default='log.txt')

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    folder_ckpt = os.path.join(BASE_DIR, 'ckpts/')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)
    
    if args.log != '-':
        sys.stdout = open(os.path.join(folder_ckpt, args.log), 'w')
    
    print(args)
    
    # load data
    folder_data = os.path.join(BASE_DIR, 'data/')
    rgb_trn = np.load(folder_data+args.xtrn)
    depth_trn = np.load(folder_data+args.ytrn)
    print('training images shape: {}'.format(rgb_trn.shape))
    print('training depth shape: {}'.format(depth_trn.shape))

    # params
    size = (9, 9, 6)
    dropout_rate = 0.3
    epochs = args.epochs
    batch_size = args.bsize
    lr = args.lr
    model_name = 'sdb_cnn'+'_dropout='+str(dropout_rate)+'_lr='+str(lr)+'_bsize='+str(batch_size)

    # create model
    model = utils.sdb_cnn(input_size=size, dropout_rate=dropout_rate)

    # call backs
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     patience=50,
                                                     verbose=1, 
                                                     min_delta=1e-6,
                                                     mode='auto')
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(folder_ckpt+model_name+'_{epoch:04d}.ckpt',
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         mode='auto',
                                                         save_weights_only=True)
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=folder_ckpt+'logs/',
                                                 histogram_freq=1)
    callbacks = [modelCheckpoint, tensorBoard]

    # compile
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = lr),
                  loss = 'mse',
                  metrics = ['mae', 'mse'])

    # show the model architecture
    model.summary()

    # fit the compiled model to some training data
    start_time = time.time()
    r = model.fit(x=rgb_trn, y=depth_trn,
              batch_size=batch_size, epochs=epochs,
              callbacks=callbacks,
              verbose=1, validation_split=0.2)
    elapsed_time = time.time() - start_time
    print('Training complete. Elapsed time: '+str(elapsed_time))

    print('{}-Done!'.format(datetime.now()))
    sys.stdout.flush()


if __name__ == "__main__":
    main()