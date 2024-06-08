from os import path as osp
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
from scipy import interpolate
from scipy.spatial.transform import Rotation
import cv2
import argparse
import open3d as o3d
import copy
import os
import sys
sys.path.append(osp.dirname(__file__))
sys.path.append(osp.dirname(osp.dirname(__file__)))
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from tqdm import tqdm
from copy import deepcopy
import json
from matplotlib import pyplot as plt

def rot_transfer(dx, dy, dtheata):
    translation = [dx, dy, 0]
    rot_zyx = [dtheata,0,0]
    return translation, rot_zyx


def load_pose(data:dict, name:str):
    obj_pose = np.eye(4)
    obj_pose[:3,:3] = data[name]['rotation']
    obj_pose[:3,3] = data[name]['location']
    return obj_pose

def inv_pose(pose:np.ndarray, inplace=False):
    if inplace:
        new_pose = pose
    else:
        new_pose = pose.copy()
    new_pose[:3,:3] = pose[:3,:3].T
    new_pose[:3,3] = -new_pose[:3,:3] @ new_pose[:3,3]
    return new_pose

class simulator(object):
    def __init__(self, data_folder:str, mesh_path:str, mesh_scale:float, zero_mean:bool=True, origin=[0,0,0]):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis

        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        if zero_mean:
            center = self.mesh.get_center()
            self.mesh.translate(-center)
        self.mesh.scale(mesh_scale,[0,0,0])
        self.mesh.translate(np.asanyarray(origin))
        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0

    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        # height_idx_max = int(np.max(height_idx))
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img

    def generateHeightMap(self, gelpad_model_path:str, pressing_height_mm:float, rel_pose:np.ndarray, raycast_argv:dict):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        # heightMap = np.zeros((psp.h,psp.w))
        tf_mesh = deepcopy(self.mesh)
        tf_mesh.rotate(rel_pose[:3,:3],
                    center=[0,0,0])
        tf_mesh.translate(rel_pose[:3,3])
        
        
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(tf_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

        # 0.025 in this code corresponds to -0.025 in visual simulation
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            **raycast_argv
        )
        ans = scene.cast_rays(rays)
        heightMap = ans['t_hit'].numpy()
        if args.debug:
            plt.imshow(heightMap)
            plt.savefig("debug_heightmap.png")

        inf_mask = np.isinf(heightMap)
        heightMap[inf_mask] = 0
        heightMap_pixel = 1000*heightMap/psp.pixmm
        min_o = np.min(heightMap_pixel[np.nonzero(heightMap_pixel)])

        heightMap_pixel = -heightMap_pixel+0.015/psp.pixmm+min_o
        heightMap_pixel[inf_mask] = 0
        min_o = np.min(heightMap_pixel[np.nonzero(heightMap_pixel)])
        heightMap_pixel[inf_mask] = min_o
        max_o = np.max(heightMap_pixel)
        # heightMap_pixel = -(heightMap_pixel - max_o)
        # heightMap_pixel[inf_mask] = 0

        # mask_map = mask_u & mask_v & mask_z
        # heightMap[vv[mask_map],uu[mask_map]] = self.vertices[mask_map][:,2]/psp.pixmm

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm
        gel_map_org = copy.deepcopy(gel_map)

        # shift the gelpad to interact with the object
        # max_o = np.max(heightMap_pixel)
        gel_map = -1 * gel_map + (max_g + max_o - pressing_height_pix)
        # get the contact area
        contact_mask = gel_map<heightMap_pixel
        # contact_mask = contact_mask & (~inf_mask)
        # heightMap = gel_map_org + heightMap - pressing_height_pix

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask] = heightMap_pixel[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]
        if args.debug:
            plt.imshow(contact_mask)
            plt.savefig("debug_mask.png")
            plt.imshow(zq)
            plt.savefig("debug_zq.png")

        return zq, gel_map, contact_mask

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        contact_height = zq - gel_map

        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        return GD1

    def generate_normals(self,height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_folder", default="objects/jar.stl")
    parser.add_argument("--obj_name",type=str,default='jar')
    parser.add_argument("--calib_dir", default="calibs")
    parser.add_argument("--gelmap",type=str,default="calibs/gelmap5.npy")
    parser.add_argument("--pose_dir",type=str,default="data/jar/pose")
    parser.add_argument("--output_dir",type=str,default="data/jar/")
    parser.add_argument('--raycast_file', type=str, default="raycast_para2/jar.json")
    parser.add_argument("--debug",type=bool,default=True)
    args = parser.parse_args()
    raycast_data = json.load(open(args.raycast_file,'r'))
    press_depth = raycast_data['depth']
    transfer_data = raycast_data['transfer']
    # gelpad_model_path = osp.join( '..', 'calibs', 'dome_gel.npy')
    gelpad_model_path = args.gelmap
    sim = simulator(args.calib_dir, args.mesh_folder, transfer_data['scale'], transfer_data['zero_mean'], transfer_data['origin'])
    pose_files = sorted(os.listdir(args.pose_dir))
    t_mask_dir = os.path.join(args.output_dir,'t_mask')
    t_img_dir = os.path.join(args.output_dir,'t_img')
    os.makedirs(t_mask_dir,exist_ok=True)
    os.makedirs(t_img_dir,exist_ok=True)
    for index, pose_file in tqdm(enumerate(pose_files),total=len(pose_files)):
        pose = json.load(open(os.path.join(args.pose_dir, pose_file),'r'))
        obj_pose = load_pose(pose, args.obj_name)
        tac_left_pose = load_pose(pose, 'Fingertip.L')
        obj_pose[:3,:3] = np.eye(3)
        obj_rel_pose = inv_pose(tac_left_pose) @ obj_pose
        # generate height map
        height_map, gel_map, contact_mask = sim.generateHeightMap(gelpad_model_path, press_depth, obj_pose, raycast_data['ray_casting'])
        # approximate the soft deformation
        # contact_height = height_map - gel_map
        heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
        contact_mask_save = np.zeros(contact_mask.shape, dtype=np.uint8)
        contact_mask_save[contact_mask>0] = 255
        cv2.imwrite(osp.join(t_mask_dir, "%04d.png"%index), contact_mask_save)
        # simulate tactile images
        sim_img, _ = sim.simulating(heightMap, contact_mask, contact_height, shadow=False)

        # sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
        # file_name = '%04d'%index
        # img_savePath = osp.join('..', 'results', 'wavecup', file_name+'_sim.jpg')
        # shadow_savePath = osp.join('..', 'results','wavecup', file_name+'_shadow.jpg')
        # height_savePath = osp.join('..', 'results','wavecup', file_name+'_height.npy')
        cv2.imwrite(osp.join(t_img_dir, '%04d.png'%index), sim_img)
        # np.save(height_savePath, heightMap)
