import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error,
                             median_absolute_error,
                             r2_score)


# evaluation overall
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
path = os.path.join(ROOT_DIR, 'data/')
model_name = 'sdb_cnn_dropout=0.3_lr=0.0001_bsize=512_0300.ckpt'
fpred = 'depth_pred_190108_'+model_name+'.npy'
ftest = 'depth_tst_190108.npy'
y_test = np.load(path+ftest)[:,0]
y_pred = np.load(path+fpred)[:,0]

df_eval = pd.DataFrame({'test': y_test, 'pred': y_pred, 'diff': np.abs(y_test-y_pred)})
df_eval = df_eval.loc[(df_eval.test < 0.0) & (df_eval.test >= -20.0)]
df_eval = df_eval.dropna()
rmse = np.sqrt((df_eval['diff']**2).mean())
mae = mean_absolute_error(df_eval['test'], df_eval['pred'])
medAE = median_absolute_error(df_eval['test'], df_eval['pred'])
r2 = r2_score(df_eval['pred'], df_eval['test'])
mini = df_eval['diff'].min()
maxi = df_eval['diff'].max()
print(f'rmse: {rmse:.2f}, mae: {mae:.2f}, r2: {r2:.2f}')
print(f'medAE: {medAE:.2f}, min: {mini:.2f}, max: {maxi:.2f}')

# tvu
order = '2' #option: special/1/2
if order == 'special':
    a = 0.25
    b = 0.0075
elif order == '1':
    a = 0.5
    b = 0.013
else:
    a = 1
    b = 0.023

df_eval = pd.DataFrame({'test': y_test, 'pred': y_pred, 'diff': np.abs(y_test-y_pred)})
df_eval = df_eval.loc[(df_eval.test < 0.0) & (df_eval.test >= -20.0)]
d = df_eval.to_numpy()
count=0
for i in range(len(d)):
    tvu = np.sqrt(a**2 + (b*d[i,0])**2)
    if d[i,2] <= tvu:
        count += 1
cl = count / len(d)
print(f'tvu: {cl:.2f}')

# plot
plt.scatter(df_eval['test'], df_eval['pred'])
plt.xlabel('True Values [Z]')
plt.ylabel('Predictions [Z]')
plt.axis('equal')
plt.axis('square')
plt.xlim([-25,4])
plt.ylim([-25,4])
_ = plt.plot([-100, 100], [-100, 100])
out_dir = os.path.join(ROOT_DIR, 'figs/')
plt.savefig(out_dir+'pred_v_ref.png')

# evaluation per depth range
start = 0.0
end = -21.0
rng = np.arange(start, end, -1.0)
for i, d in enumerate(rng):
    if rng[i] != rng[-1]:
        df_eval = pd.DataFrame({'test': y_test, 'pred': y_pred, 'diff': np.abs(y_test-y_pred)})
        df_eval = df_eval.loc[(df_eval.test < rng[i]) & (df_eval.test >= rng[i+1])]
        rmse = np.sqrt((df_eval['diff']**2).mean())
        mae = mean_absolute_error(df_eval['test'], df_eval['pred'])
        d = df_eval.to_numpy()
        count=0
        for ix in range(len(d)):
            tvu = np.sqrt(a**2 + (b*d[ix,0])**2)
            if d[ix,2] <= tvu:
                count += 1
        cl = count / len(d)
        mini = df_eval['diff'].min()
        maxi = df_eval['diff'].max()
        print(f'range {rng[i]} to {rng[i+1]}    rmse: {rmse:.2f}, mae: {mae:.2f}, min: {mini:.2f}, max: {maxi:.2f}, tvu: {cl:.2f}')

