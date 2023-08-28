import open3d as o3d
import argparse
import numpy as np
import os
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius",type=float,default=2)
    parser.add_argument("--polar_resol",type=float,default=0.05)
    parser.add_argument("--azimuth_resol",type=float,default=0.08)
    parser.add_argument("--save_path",type=str,default="semi-sphere.pcd")
    return parser.parse_args()

if __name__ == "__main__":
    args = options()
    polar_units = np.arange(-np.pi/2, 0, args.polar_resol)
    azimuth_units = np.arange(-2*np.pi, 2*np.pi, args.azimuth_resol)
    theta, phi = np.meshgrid(polar_units, azimuth_units)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)
    tmp = args.radius * np.sin(theta)
    X = tmp * np.cos(phi)
    Y = tmp * np.sin(phi)
    Z = args.radius * np.cos(theta)
    radii = [0.1,0.2,0.4,0.8]
    Pt_arr = np.hstack((X[:,None],Y[:,None],Z[:,None]))
    Pt = o3d.geometry.PointCloud()
    Pt.points = o3d.utility.Vector3dVector(Pt_arr)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(__file__),"objects",args.save_path),Pt,write_ascii=True)
            