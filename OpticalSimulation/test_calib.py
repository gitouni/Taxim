from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import cv2
import argparse
import numpy as np
# import matplotlib
# matplotlib.use("agg")
from matplotlib import pyplot as plt
from utils import fast_poisson

os.chdir(os.path.dirname(__file__))
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="../results/RFreg_att.joblib")
    parser.add_argument("--test_sim",type=str,default="../results/asim/0007.jpg")
    parser.add_argument("--gt_file",type=str,default="../results/aheight/0007.npz")
    parser.add_argument("--max_margin",type=float,default=1.25)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    tree:RandomForestRegressor = joblib.load(args.model_path)
    sim_img = cv2.imread(args.test_sim)  # (H, W ,3)
    H, W = sim_img.shape[:2]
    gt_data = np.load(args.gt_file)
    dzdx = gt_data['dzdx']
    dzdy = gt_data['dzdy']
    gt_heigt = gt_data['height']
    gt_heigt[np.logical_not(gt_data['contact_mask'])] = 0
    dzdx_max = dzdx.max()
    dzdy_max = dzdy.max()
    dzdx_min = dzdx.min()  # < 0
    dzdy_min = dzdy.min()  # < 0
    h_max = gt_heigt.max()
    pred_grad = tree.predict(sim_img.reshape(-1, 3)).reshape(H, W , 2)
    pred_height = fast_poisson(dzdx, dzdy)
    
    plt.figure()
    ax = plt.subplot(2,3,1)
    ax.set_title("ground-truth dzdx")
    ax.imshow(dzdx, vmax=dzdx_max * args.max_margin, vmin=dzdx_min * args.max_margin)
    
    ax = plt.subplot(2,3,2)
    ax.set_title("ground-truth dzdy")
    ax.imshow(dzdy, vmax=dzdy_max * args.max_margin, vmin=dzdy_min * args.max_margin)
    
    ax = plt.subplot(2,3,3)
    ax.set_title("ground-truth height")
    ax.imshow(gt_heigt, vmax=h_max * args.max_margin, vmin=0)
    
    ax = plt.subplot(2,3,4)
    ax.set_title("predicted dzdx")
    ax.imshow(pred_grad[...,0], vmax=dzdx_max * args.max_margin, vmin=dzdx_min * args.max_margin)
    
    ax = plt.subplot(2,3,5)
    ax.set_title("predicted dzdy")
    ax.imshow(pred_grad[...,1], vmax=dzdx_max * args.max_margin, vmin=dzdx_min * args.max_margin)

    ax = plt.subplot(2,3,6)
    ax.set_title("predicted height")
    ax.imshow(pred_height)
    plt.tight_layout()
    
    plt.show()