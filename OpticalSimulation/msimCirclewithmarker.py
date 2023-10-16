import os
from os import path as osp
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
# from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
from scipy.stats.qmc import Halton
import cv2
import argparse
# import open3d as o3d
# from utils import getDomeHeightMap
from tqdm import tqdm
os.chdir(osp.dirname(__file__))
import sys
sys.path.append("../")
# from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from MarkerMotionSimulation.compose.superposition import SuperPosition, fill_blank
os.chdir(osp.dirname(__file__))
def options():
    parser = argparse.ArgumentParser()
    obj_parser = parser.add_argument_group()
    obj_parser.add_argument("--obj", nargs='?', default='semi-sphere',
                        help="Name of Object to be tested, supported_objects_list = [square, cylinder6]")
    obj_parser.add_argument("--radius", default=2.832, type=float, help="unit: mm")
    obj_parser.add_argument("--data_folder",type=str,default=osp.join("..", 'calibs'))
    obj_parser.add_argument('--max_depth', default=2.832, type=float, help='Indetation depth into the gelpad.')
    obj_parser.add_argument('--min_depth',default= 1.0, type=float)
    obj_parser.add_argument("--max_margin_ratio",type=float,default=0.6)
    obj_parser.add_argument("--sample_num",type=int,default=10)
    obj_parser.add_argument("--seed",type=int,default=0)
    obj_parser.add_argument("--suffix",type=str,default=".pcd")
    obj_parser.add_argument("--zero_mean",type=bool,default=False)
    marker_parser = parser.add_argument_group()
    marker_parser.add_argument("--marker_init_pos", type=int, nargs=2, default=[0,0])
    marker_parser.add_argument("--marker_rows",type=int, default=8, help="number of intervals")
    marker_parser.add_argument("--marker_cols",type=int, default=6, help="number of intervals")
    marker_parser.add_argument("--marker_pix_distance", type=int, default=80)
    marker_parser.add_argument("--marker_radius",type=int, default=8)
    io_parser = parser.add_argument_group()
    io_parser.add_argument("--result_basedir",type=str,default="../results")
    io_parser.add_argument("--sim_dir",type=str,default="asim_marker")
    io_parser.add_argument("--height_dir",type=str,default="aheight_marker")
    return parser.parse_args()

class simulator(object):
    def __init__(self, args:argparse.Namespace):
        """construct the gelsight simulator

        Args:
            args (argparse.Namespace): args
        """
        # read in object's ply file
        # object facing positive direction of z axis

        # f = open(objPath)
        # lines = f.readlines()
        # self.verts_num = int(lines[3].split(' ')[-1])
        # verts_lines = lines[10:10 + self.verts_num]
        # self.vertices = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])

        # polytable
        polycalib_path = osp.join(args.data_folder, "polycalib.npz")
        self.calib_data = CalibData(osp.abspath(polycalib_path))
        self.radius = args.radius
        # raw calibration data
        datapack_path = osp.join(args.data_folder, "dataPack.npz")
        data_file = np.load(datapack_path,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()
        # marker motion data
        fem_calib_path = osp.join(args.data_folder, "femCalib.npz")
        self.fem_super = SuperPosition(osp.abspath(fem_calib_path))
        self.domeMap = np.load(osp.join(args.data_folder, 'dome_gel.npy'))
        marker_x0, marker_y0 = args.marker_init_pos
        marker_xmax = marker_x0 + args.marker_rows * args.marker_pix_distance
        marker_ymax = marker_y0 + args.marker_cols * args.marker_pix_distance
        marker_x, marker_y = np.meshgrid(np.arange(marker_x0, marker_xmax + 1e-4, args.marker_pix_distance),
                                         np.arange(marker_y0, marker_ymax + 1e-4, args.marker_pix_distance))
        self.marker_pos_arr = np.concatenate((marker_x[...,None], marker_y[...,None]),axis=-1).reshape(-1,2).astype(np.int32)
        self.marker_radius = args.marker_radius
        # shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(args.data_folder, "shadowTable.npz"),allow_pickle=True)
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
        convEachDim = lambda in_img : gaussian_filter(in_img, kscale)

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
        # h,w,ch = f0.shape
        # pixcount = h*w

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
        dzdx: gradient dz/dx of the image
        dzdy: gradient dz/dy of the image
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)
        dzdx = grad_mag * np.cos(grad_dir)
        dzdy = grad_mag * np.sin(grad_dir)
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

        # # add shadow
        # cx = psp.w//2
        # cy = psp.h//2

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
            frame:np.ndarray = sim_img_r[:,:,c].copy()
            # frame_back = sim_img_r[:,:,c].copy()
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
        
        
        
        return sim_img, shadow_sim_img, dzdx, dzdy

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, du, dv):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        du, dv: shift of the object (pixel)
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # load dome-shape gelpad model
        assert(pressing_height_mm <= self.radius), "pressing depth ({}) > radius ({})".format(pressing_height_mm, self.radius)
        
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        heightMap = np.zeros((psp.h,psp.w))
        mesh_u, mesh_v = np.meshgrid(np.arange(psp.w), np.arange(psp.h))
        mesh_u = mesh_u.reshape(-1).astype(np.int32)
        mesh_v = mesh_v.reshape(-1).astype(np.int32)
        mesh_x = (mesh_u - psp.w//2) * psp.pixmm
        mesh_y = (mesh_v - psp.h//2) * psp.pixmm
        dx = du * psp.pixmm
        dy = dv * psp.pixmm
        mask = (mesh_x - dx)**2 + (mesh_y - dy)**2 <= self.radius**2
        heightMap[mesh_v[mask], mesh_u[mask]] = np.sqrt(self.radius**2 - (mesh_x[mask] - dx) ** 2 - (mesh_y[mask] - dy) ** 2)
        heightMap[heightMap < psp.minimum_contact] = 0
        heightMap /= psp.pixmm
        max_g = np.max(gel_map)
        # min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]
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
        grad_dir = np.zeros((h-2,w-2))  # set the direction of invalid grads to 0
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')

if __name__ == "__main__":
    args = options()
    assert(args.radius >= args.max_depth)
    data_folder = osp.join(osp.join( "..", "calibs"))
    gelpad_model_path = osp.join( '..', 'calibs', 'gelmap5.npy')
    os.makedirs(osp.join(args.result_basedir, args.sim_dir), exist_ok=True)
    # os.makedirs(osp.join("..","results","ashadow"), exist_ok=True)
    os.makedirs(osp.join(args.result_basedir, args.height_dir), exist_ok=True)
    sim = simulator(args)
    sampler = Halton(3,seed=args.seed)
    rand_data = sampler.random(args.sample_num)
    depth_list = (args.max_depth - args.min_depth) * rand_data[:,0] + args.min_depth
    du_list = np.array(psp.w * (rand_data[:,1] - 0.5) * args.max_margin_ratio, dtype=np.int32)
    dv_list = np.array(psp.h * (rand_data[:,2] - 0.5) * args.max_margin_ratio, dtype=np.int32)
    height_map, gel_map, contact_mask = sim.generateHeightMap(gelpad_model_path, 0, 0, 0)
    # approximate the soft deformation
    heightMap, contact_mask, contact_height = sim.deformApprox(0, height_map, gel_map, contact_mask)
    # simulate tactile images
    sim_img, _, dzdx, dzdy = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
    ref_marker_img = np.zeros((psp.h, psp.w), dtype=np.uint8)  # uint8
    for marker_pos in sim.marker_pos_arr:
        cv2.circle(ref_marker_img, marker_pos, radius=sim.marker_radius, color=255, thickness=-1)  # solid
    ref_binary_marker = ref_marker_img > 0
    marker_coords = np.nonzero(ref_marker_img)  # (y_arr, x_arr)
    sim_img[ref_binary_marker,:] = np.array([0,0,0],dtype=np.int32)  # add deformed markers
    img_savePath = osp.join(args.result_basedir, args.sim_dir, '%04d.jpg'%(0))
    # shadow_savePath = osp.join('..', 'results', 'ashadow', '%04d.jpg'%(0))
    height_savePath = osp.join(args.result_basedir, args.height_dir, '%04d.npz'%(0))
    cv2.imwrite(img_savePath, sim_img)
    np.savez(height_savePath, height=height_map*psp.pixmm, dzdx=dzdx, dzdy=dzdy, contact_mask=contact_mask)
    for i, (depth, du, dv) in tqdm(enumerate(zip(depth_list, du_list, dv_list)), total=len(du_list)):
        # generate height map
        height_map, gel_map, contact_mask = sim.generateHeightMap(gelpad_model_path, depth, du, dv)
        # approximate the soft deformation
        heightMap, contact_mask, contact_height = sim.deformApprox(depth, height_map, gel_map, contact_mask)
        # simulate tactile images
        sim_img, shadow_sim_img, dzdx, dzdy = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
        local_deform = (0,0,depth)
        resultMap = sim.fem_super.compose_sparse_insert(local_deform, gel_map, contact_mask)
        xbias = fill_blank(resultMap[0,...])
        ybias = fill_blank(resultMap[1,...])
        marker_xarr = np.array(marker_coords[1] + xbias[ref_binary_marker],dtype=np.int32)  # floor integar approx
        marker_yarr = np.array(marker_coords[0] + ybias[ref_binary_marker],dtype=np.int32)  # floor integar approx
        rev = (0 <= marker_xarr) * (marker_xarr < psp.w) * (0 <= marker_yarr) * (marker_yarr < psp.h)
        sim_img[marker_yarr[rev], marker_xarr[rev],:] = np.array([0,0,0],dtype=np.int32)  # add deformed markers
        marker_xarr = np.array(marker_coords[1] + xbias[ref_binary_marker] + 0.5,dtype=np.int32)  # ceil integar approx
        marker_yarr = np.array(marker_coords[0] + ybias[ref_binary_marker] + 0.5,dtype=np.int32)  # ceil integar approx
        rev = (0 <= marker_xarr) * (marker_xarr < psp.w) * (0 <= marker_yarr) * (marker_yarr < psp.h)
        sim_img[marker_yarr[rev], marker_xarr[rev],:] = np.array([0,0,0],dtype=np.int32)  # add deformed markers
        img_savePath = osp.join(args.result_basedir, args.sim_dir, '%04d.jpg'%(i+1))
        # shadow_savePath = osp.join('..', 'results', 'ashadow', '%04d.jpg'%(i+1))
        height_savePath = osp.join(args.result_basedir, args.height_dir, '%04d.npz'%(i+1))
        cv2.imwrite(img_savePath, sim_img)
        # cv2.imwrite(shadow_savePath, shadow_sim_img)
        np.savez(height_savePath, height=height_map*psp.pixmm, dzdx=dzdx, dzdy=dzdy, contact_mask=contact_mask)
