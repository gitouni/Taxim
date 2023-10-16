from sklearn.ensemble import RandomForestRegressor
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(__file__))
import joblib

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",type=str,default="../results/asim")
    parser.add_argument("--height",type=str,default="../results/aheight")
    parser.add_argument("--n_est",type=int,default=30)
    parser.add_argument("--max_leaf",type=int,default=None)
    parser.add_argument("--resample_bg",type=int,default=False)
    parser.add_argument("--bg_weight", type=float, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    tree = RandomForestRegressor(n_estimators=args.n_est, max_leaf_nodes=args.max_leaf, n_jobs=-1)
    sim_files = list(sorted(os.listdir(args.sim)))
    height_files = list(sorted(os.listdir(args.height)))
    ref_file = os.path.join(args.sim, sim_files[0])
    ref_img = cv2.imread(ref_file)  # H, W ,3
    h, w = ref_img.shape[:2]
    ref_pixel = ref_img.reshape(-1,3)
    pixels = [ref_img.reshape(-1,3)]  # (N, 3)
    grads = [np.zeros((ref_pixel.shape[0], 2))]  # (N, 2)
    for sim_file, height_file in tqdm(zip(sim_files[1:], height_files[1:]), total=len(sim_files)-1, desc="loading data"):
        sim_file = os.path.join(args.sim, sim_file)
        height_file = os.path.join(args.height, height_file)
        sim_img = cv2.imread(sim_file)  # H, W, 3
        data = np.load(height_file)
        mask = data['contact_mask']
        dzdx = data['dzdx']
        dzdy = data['dzdy']
        pixels.append(sim_img[mask].reshape(-1,3))  # (N, 3)
        grads.append(np.stack((dzdx[mask].reshape(-1), dzdy[mask].reshape(-1)), axis=1))  # (N ,2)
    pixels = np.concatenate(pixels, axis=0)  # list of (N,3) -> (kN,3)
    grads = np.concatenate(grads, axis=0) # list of (N,2) -> (kN,2)
    N = pixels.shape[0]
    bgN = h * w
    trN = int(0.75*(N - bgN))
    print("number of samples: {}, {} for background, {} for train, {} for validation".format(N, bgN, trN, N-trN-bgN))
    random_indices = np.random.permutation(N - bgN)
    tr_id = random_indices[:trN] + bgN
    val_id = random_indices[trN:] + bgN
    tr_id = np.concatenate((tr_id, np.arange(bgN,dtype=np.int32)))
    val_id = np.concatenate((val_id, np.arange(bgN,dtype=np.int32)))
    print("Start training RandomForestRregressor")
    if args.resample_bg:
        sample_weight = np.ones_like(tr_id, dtype=np.float32)
        sample_weight[-bgN:] = args.bg_weight
        tree.fit(pixels[tr_id], grads[tr_id], sample_weight=sample_weight)
    else:
        tree.fit(pixels[tr_id], grads[tr_id])
    print("Training finished. Start verifying...")
    r2 = tree.score(pixels[val_id], grads[val_id])
    print("Validation dataset R2: {}".format(r2))
    joblib.dump(tree, "../results/RFreg_att.joblib") # save format recommended by the official