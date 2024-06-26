import numpy as np
import torch
from typing import List, Tuple, Union
import sys
import time
import math
import os 
import torch.distributed as dist
import datetime
import builtins
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch.nn.functional as F
from src.common import rotation_conversions
try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def transform_to_frame(xyz, pose):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
    
    Args:
        xyz: Nx3
        pose: Mx4x4
    
    Returns:
        MxN mask 
        transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations)
    """
    # Get Frame Camera Pose
    xyz_hom = to_hom(xyz)
    # world_to_camera = torch.inverse(pose) # to speed up we can use here the pose inverted
    # M x 4 x 4 , M x N x 4
    xyz_hom = xyz_hom.unsqueeze(0).repeat(pose.size(0),1,1)
    world_to_camera = pose.float()
    xyz_camera = torch.bmm(xyz_hom, world_to_camera.transpose(-2,-1))[...,:3] # MxNx3

    depth_mask = (xyz_camera[...,2:] > 0) # .nonzero() # z > 0 MxN
    # depth_order = torch.argsort(xyz_camera[...,2],dim=-1,descending=False)

    return xyz_camera, depth_mask #, depth_order

def vis_coordinate(cfg, box_coord, global_step, save_path, scene_name):
    """
	 box_coord # H x W x Z x 3
     # we draw the changes heatmap, larger value means the area have larger change
     # we show density map, to describe the area with more points the one with less 
     # above we show 3D or 2D, need a good way for visualize
	 """
    plt.close("all")
    init_box_coord = make_3D_coordinates(cfg.net.volume_range, cfg.net.volume_size) 
    original_color_distance = torch.norm(init_box_coord, dim=-1)

    points = box_coord.detach().cpu().numpy().reshape(-1, 3)
    np.save("{}/{}_learn_coord.npy".format(save_path,global_step), points)
    if not os.path.exists("{}/init_coord.npy".format(save_path)):
        np.save("{}/init_coord.npy".format(save_path), init_box_coord.detach().cpu().numpy().reshape(-1, 3))

    flat_original_color_distance = original_color_distance.detach().cpu().numpy().reshape(-1)

    # Normalize distances for coloring
    distances_normalized = (flat_original_color_distance  / flat_original_color_distance.max() )
    volume_fig_size = int(24 *cfg.net.volume_size[0]//16)
    fig_3D = plt.figure(figsize=(volume_fig_size,volume_fig_size))
    ax = fig_3D.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances_normalized, cmap='viridis')

    # Add a color bar
    # plt.colorbar(sc, ax=ax, label='Distance to origin')
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('learnable coordinate Visualization')
    save_path = os.path.join(save_path, scene_name)
    png_fname = "{}/{}_coord.png".format(save_path,global_step)
    plt.savefig(png_fname,dpi=75)
    plt.close("all")


    # 2D slice
    images_fig_size = int(24*cfg.net.volume_size[0]//16)
    fig_2D = plt.figure(figsize=(24,24))
    ax = fig_2D.add_subplot(111)
    slice_idx = cfg.net.volume_size[2] // 2
    slice_points = box_coord[:,:,slice_idx].detach().cpu().numpy().reshape(-1, 3)
    slice_original_color_distance = original_color_distance[:,:,slice_idx].detach().cpu().numpy().reshape(-1)
    # original_color_distance.reshape(cfg.net.volume_size[0], cfg.net.volume_size[1], cfg.net.volume_size[2])[:,:,slice_idx].reshape(-1)
    slice_distances_normalized = (slice_original_color_distance / slice_original_color_distance.max())
    sc = ax.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_distances_normalized, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Distance to origin')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('learnable coordinate Visualization 2D central slice')
    png_fname = "{}/{}_coord_2D.png".format(save_path,global_step)
    plt.savefig(png_fname,dpi=75)

    return fig_3D, fig_2D

    


def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = pose #torch.inverse.invert(pose)
    # from opengl to opencv
    # pose_inv = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],dtype=pose.dtype,device=pose.device)@pose
    return X_hom@pose_inv.transpose(-1,-2)

def get_camera_mesh(pose,depth=0.2/5):
    vertices = torch.tensor([[-0.5,-0.5,1],
                             [0.5,-0.5,1],
                             [0.5,0.5,1],
                             [-0.5,0.5,1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = cam2world(vertices[None],pose)
    # print("pose", pose.size())
    vertices = vertices[...,:3]
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices,faces,wireframe

def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged

def vis_cameras(poses_pred, poses_gt, epoch, save_path, scene_name, convention='opengl'):
    plt.close("all")
    if convention == 'opengl':
        poses_pred[:,:,:3,1:3] *= -1
        poses_gt[:,:,:3,1:3] *= -1
    elif convention == 'opencv':
        pass
    # set up plots
    # poses [poses_gt, poses_pred]
    save_path = os.path.join(save_path, scene_name)
    os.makedirs(save_path,exist_ok=True)
    poses_pred = poses_pred[0].detach().cpu() # use first batch / scene
    poses_gt = poses_gt[0].detach().cpu()

    # save the camera poses as npy
    if not os.path.exists("{}/GT.npy".format(save_path)):
        np.save("{}/GT.npy".format(save_path),poses_gt.numpy())
    np.save("{}/{}.npy".format(save_path,epoch),poses_pred.numpy())
    _,_,cam = get_camera_mesh(poses_pred)
    _,_,cam_ref = get_camera_mesh(poses_gt)
    fig = plt.figure(figsize=(10,10))

    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    ax.set_title("{} global_step {}".format(scene_name, epoch),pad=0)
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3/5,3/5),y=(-3/5,3/5),z=(-3/5,2.4/5)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    if not os.path.exists("{}/GT.png".format(save_path)):
        png_fname = "{}/GT.png".format(save_path)
        plt.savefig(png_fname,dpi=75)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    png_fname = "{}/{}.png".format(save_path,epoch)
    # print("png_fname", png_fname)
    plt.savefig(png_fname,dpi=75)
    # clean up
    return fig



    


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(args):
    if args.no_env:
        pass
    elif args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        # flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def rotation_distance_torch(pred,gt):
    # Compute the relative rotation matrix R
    B, S = pred.shape[:2]
    pred = pred.reshape(-1,3,3)
    gt = gt.reshape(-1,3,3)
    R = torch.bmm(pred.transpose(-2, -1), gt)
    # Calculate the trace of R
    trace_R = torch.einsum('bii->b', R)
    # Compute the rotation angle in radians
    # Clamp the value to prevent numerical issues that might lead to complex numbers
    cos_theta = (trace_R - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)  # Clamp values to ensure they are within the valid range for acos
    theta = torch.acos(cos_theta)
    theta = theta.reshape(B,S,1)
    return theta  # or return theta for the angle in radians
   

def rotation_distance_euler(pred,gt):
    # Compute the relative rotation matrix R
    B, S = pred.shape[:2]
    pred = pred.reshape(-1,3,3)
    gt = gt.reshape(-1,3,3)

    euler_gt = rotation_conversions.matrix_to_euler_angles(gt,['X','Y','Z'])
    euler_pred = rotation_conversions.matrix_to_euler_angles(gt,['X','Y','Z'])

    euler_diff = (euler_gt - euler_pred).abs().mean()
    return torch.rad2deg(euler_diff)




def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f




def positional_encoding(input,L): # [B,...,N]
    shape = input.shape
    freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)*np.pi # [L]
    spectrum = input[...,None]*freq # [B,...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
    return input_enc


def pixels_to_patch(pixels,shifts):
    """
    pixels: N x 2
    output: N x patch H x patch width x 2
    """
    patch_height, patch_width = shifts.shape[:2]
    patch_pixels = pixels.reshape(-1,1,1,2).repeat(1,patch_height,patch_width,1)

    patch_pixels_activ = patch_pixels
    # patch_pixels_activ = torch.nn.functional.tanh(patch_pixels)

    assert patch_pixels_activ.shape[-3:] == shifts.shape[-3:]
    patch_pixels_shift = patch_pixels_activ+shifts

    return patch_pixels_shift

# use only for our cifar mlps
def normalized_coord_to_rgb_with_given_weights(input, weights_batch, bias_batch):
    output = [input.transpose(-2, -1)]
    
    for layer_i, (weight_i, bias_i) in enumerate(zip(weights_batch,bias_batch )):
        weight_i.requires_grad = False 
        bias_i.requires_grad = False 
        output_i = torch.matmul(weight_i, output[layer_i]) + bias_i.unsqueeze(-1)

        if layer_i <2:
            output_i = torch.nn.functional.relu(output_i)
        output.append(output_i)

    return output[-1]


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)

def make_3D_coordinates(box_coord, volume_size):
    xmin, xmax, ymin, ymax, zmin, zmax = box_coord
    H,W,Z = volume_size
    # print("x min max", xmin, xmax)
    # print("y min max", ymin, ymax)
    # print("z min max", zmin, zmax)
    # print("HWZ", H,W,Z)
    x_coords = torch.linspace(start=xmin, end=xmax, steps=W)
    y_coords = torch.linspace(start=ymin, end=ymax, steps=H)
    z_coords = torch.linspace(start=zmin, end=zmax, steps=Z)
    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords)
    coordinates = torch.stack([X, Y, Z], dim=-1)
    return coordinates   

def make_3D_coordinates_learnable(box_coord, volume_size):
    xmin, xmax, ymin, ymax, zmin, zmax = box_coord
    coord_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    scale_factor = [(xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2]
    print("coord_center", coord_center)
    print("scale_factor", scale_factor)

    H,W,Z = volume_size

    x_coords = torch.asin(torch.linspace(-1, 1, volume_size[0]))
    y_coords = torch.asin(torch.linspace(-1, 1, volume_size[1]))
    z_coords = torch.asin(torch.linspace(-1, 1, volume_size[2]))

    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords)
    coordinates = torch.stack([X, Y, Z], dim=-1).detach().cpu().requires_grad_(False)
    return coordinates, torch.tensor(coord_center).view(1,3), torch.tensor(scale_factor).view(1,3)



def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1],layers[1:]))

def get_normalized_pixel_grid(H, W):
    y_range = ((torch.arange(H,dtype=torch.float32)+0.5)/H*2-1)*(H/max(H,W))
    x_range = ((torch.arange(W,dtype=torch.float32)+0.5)/W*2-1)*(W/max(H,W))
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    return xy_grid


def get_bilinear_weights_loc(H,W,coords):
    """
    coords N x 2, 0-1
    return N 4 x 2, N 4
    """
    # print("coords", coords.size())
    x = (0.5+0.5*coords[:, 0]) * (W - 1)
    y = (0.5+0.5*coords[:, 1]) * (H - 1)

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Calculate the distances of each coordinate from the pixels
    lx = x - x0.float()
    ly = y - y0.float()
    hx = 1 - lx
    hy = 1 - ly

    # Calculate the weights for each of the four corners
    w_tl = hx * hy
    w_tr = lx * hy
    w_bl = hx * ly
    w_br = lx * ly

    # get pixel of 4 corners
    x0_normalize = 2*(x - lx + 0.5) / W - 1  # make it derivativble
    x1_normalize =2*(x - lx + 1.5) / W - 1
    y0_normalize = 2*(y - ly + 0.5) / H - 1 
    y1_normalize = 2*(y - ly + 1.5) / H - 1

    ptl = torch.stack([x0_normalize, y0_normalize],dim=-1).unsqueeze(1)
    # print("ptl", ptl)
    ptr = torch.stack([x1_normalize, y0_normalize],dim=-1).unsqueeze(1)
    pbl = torch.stack([x0_normalize, y1_normalize],dim=-1).unsqueeze(1)
    ptr = torch.stack([x1_normalize, y1_normalize],dim=-1).unsqueeze(1)

    p = torch.cat([ptl,ptr,pbl,ptr], dim = 1)
    w = torch.cat([w_tl.unsqueeze(1), w_tr.unsqueeze(1), w_bl.unsqueeze(1), w_br.unsqueeze(1)], dim = -1 )
    return p, w

def get_bilinear_weights_loc_siren(H,W,coords,coord_range: Union[Tuple[int], List[int]] = (-1, 1),):
    """
    coords N x 2, 0-1
    return N 4 x 2, N 4
    """
    # remap to 0 - 32 so easy to calculate integer
    x_coordinates = np.linspace(coord_range[0], coord_range[1], W)
    y_coordinates = np.linspace(coord_range[0], coord_range[1], H)
    x_remap_weights = x_coordinates[-1] - x_coordinates[0]
    x_remap_bias = x_coordinates[0]
    y_remap_weights = y_coordinates[-1] - y_coordinates[0]
    y_remap_bias = y_coordinates[0]

    x = (x_remap_bias + coords[:, 0]) / x_remap_weights * (W - 1)
    y = (y_remap_bias + coords[:, 1]) / y_remap_weights * (H - 1)

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # Calculate the distances of each coordinate from the pixels
    lx = x - x0.float()
    ly = y - y0.float()
    hx = 1 - lx
    hy = 1 - ly

    # Calculate the weights for each of the four corners
    w_tl = hx * hy
    w_tr = lx * hy
    w_bl = hx * ly
    w_br = lx * ly

    # get pixel of 4 corners
    x0_normalize = (x_remap_weights * (x - lx) / W - 1  ) - x_remap_bias
    x1_normalize = (x_remap_weights * (x - lx + 1) / W - 1  ) - x_remap_bias
    y0_normalize = (y_remap_weights * (y - ly ) / H - 1  ) - y_remap_bias 
    y1_normalize = (y_remap_weights * (y - ly + 1 ) / H - 1  ) - y_remap_bias 
    

    ptl = torch.stack([x0_normalize, y0_normalize],dim=-1).unsqueeze(1)
    # print("ptl", ptl)
    ptr = torch.stack([x1_normalize, y0_normalize],dim=-1).unsqueeze(1)
    pbl = torch.stack([x0_normalize, y1_normalize],dim=-1).unsqueeze(1)
    ptr = torch.stack([x1_normalize, y1_normalize],dim=-1).unsqueeze(1)

    p = torch.cat([ptl,ptr,pbl,ptr], dim = 1)
    w = torch.cat([w_tl.unsqueeze(1), w_tr.unsqueeze(1), w_bl.unsqueeze(1), w_br.unsqueeze(1)], dim = -1 )
    return p, w



class StraightThroughMeshGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, H, W):
        # Create meshgrid
        y_range = ((torch.arange(H,dtype=torch.float32)+0.5)/H*2-1) # map to -1 to 1
        x_range = ((torch.arange(W,dtype=torch.float32)+0.5)/W*2-1) # map tp -1 to 1
        # print("x_range", x_range)
        Y,X = torch.meshgrid(y_range,x_range) 

        grid = torch.stack([X,Y],dim=-1).reshape(-1,2).to(input.device)
        input_expanded = (input.unsqueeze(1))  # Shape: [batch_size, 1, 2]

        # Compute distances from each input to each grid point
        distances = torch.norm(input_expanded - grid, dim=-1)  # Shape: [batch_size, H*W]

        # Find the index of the closest grid point
        _, min_indices = torch.min(distances, dim=-1)

        # Convert indices back to 2D coordinates
        closest_points = grid[min_indices]

        ctx.save_for_backward(input, closest_points)

        return closest_points

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: Just pass the gradient
        input, closest_points = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None