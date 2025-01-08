import numpy as np
import os
import os.path as osp
import random
import time
import torch
from sklearn.neighbors import KDTree
import open3d as o3d
from collections import OrderedDict



# np.random.seed(0)

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def txt2sample(path):
    
    index = ['X', 'Y', 'Z', 'Rf', 'Gf', 'Bf', 'label']
    with open(path, 'r') as f:
        lines = f.readlines()
        head = lines[0][2:].strip('\n').split(' ')
        ids = tuple([head.index(i) for i in index])
    points = np.loadtxt(path, skiprows=2, usecols = ids)   

    return points
  

def random_subsample(points, n_samples):
    """
    random subsample points when input points has larger length than n_samples 
    or add zeros for points which has smaller length than n_samples.
    """
    if points.shape[0]==0:
#         print('No points found at this center replacing with dummy')
        points = np.zeros((n_samples,points.shape[1]))
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0], n_samples, replace=False)
        points = points[random_indices, :]
    if n_samples > points.shape[0]:
        apd = np.zeros((n_samples-points.shape[0], points.shape[1]))
        points = np.vstack((points, apd))
    return points


def align_length(p0_path, p1_path, length):
    """
    output a pair of points with the same length.
    """
    p0 = txt2sample(p0_path) 
    p1 = txt2sample(p1_path)
    p0_raw_length = p0.shape[0]
    p1_raw_length = p1.shape[0]
    if p0.shape[0] != length:
        p0 = random_subsample(p0, length)
    if p1.shape[0] != length:
        p1 = random_subsample(p1, length) 
    return p0, p1, p0_raw_length, p1_raw_length

def get_errors(err):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err', err.item()),
            ])

        return errors
        
def plot_current_errors(epoch, counter_ratio, errors,vis):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        
        plot_data = {}
        plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([errors[k] for k in plot_data['legend']])
        
        vis.line(win='wire train loss', update='append',
            X = np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y = np.array(plot_data['Y']),
            opts={
                'title': 'Change Detection' + ' loss over time',
                'legend': plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            })

def save_weights(epoch,net,optimizer,save_path, model_name):
    if isinstance(net, torch.nn.DataParallel):
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            }
    else:
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
            }
    torch.save(checkpoint,os.path.join(save_path,'current_%s.pth'%(model_name)))
    if epoch % 1 == 0:
        torch.save(checkpoint,os.path.join(save_path,'%d_%s.pth'%(epoch,model_name)))

def plot_performance( epoch, performance, vis):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        plot_res = []
        plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        plot_res['X'].append(epoch)
        plot_res['Y'].append([performance[k] for k in plot_res['legend']])
        vis.line(win='AUC', update='append',
            X=np.stack([np.array(plot_res['X'])] * len(plot_res['legend']), 1),
            Y=np.array(plot_res['Y']),
            opts={
                'title': 'Testing ' + 'Performance Metrics',
                'legend': plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
        ) 

def save_cfg(cfg, path):
    mkdir(path)
    if not cfg.resume:
        if os.path.exists(os.path.join(path, 'configure.txt')):
            os.remove(os.path.join(path, 'configure.txt'))
        if os.path.exists(os.path.join(path, 'train_loss.txt')):
            os.remove(os.path.join(path, 'train_loss.txt'))
        if os.path.exists(os.path.join(path, 'val_metric.txt')):
            os.remove(os.path.join(path, 'val_metric.txt'))
        if os.path.exists(os.path.join(path, 'val_performance.txt')):
            os.remove(os.path.join(path, 'val_performance.txt')) 
        if os.path.exists(os.path.join(path, 'test_performance.txt')):
            os.remove(os.path.join(path, 'test_performance.txt'))      
        with open(os.path.join(path, 'configure.txt'), 'a') as f:
            f.write('---------------{}----------------'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write('\n')
            f.write('----------------Network and training configure-----------------')
            f.write('\n')
            for k in cfg:
                f.write(str(k)+':')
                f.write(str(cfg[k]))
                f.write('\n')

def save_prediction2(p0, p1, lb0, lb1, scores0, scores1, path, pc0_name, pc1_name, data_path):
    p0_name, p1_name = pc0_name[0], pc1_name[0]
    mkdir(path)
    p0 = p0[:, :, :3].squeeze(0).detach().cpu().numpy()
    p1 = p1[:, :, :3].squeeze(0).detach().cpu().numpy()
#     p00 = np.loadtxt(osp.join(data_path, path.split('\\')[-1], p0_name), usecols=(0, 1, 2, 6), skiprows=2)
#     p11 = np.loadtxt(osp.join(data_path, path.split('\\')[-1], p1_name), usecols=(0, 1, 2, 6), skiprows=2)
    lb0 = lb0.transpose(1,0).detach().cpu().numpy()
    lb1 = lb1.transpose(1,0).detach().cpu().numpy()
    scores0 = scores0.transpose(1,0).detach().cpu().numpy()
    scores1 = scores1.transpose(1,0).detach().cpu().numpy()

    p0 = np.hstack((p0, lb0, scores0))
    if tcfg.remove_plane:
        plane0 = np.loadtxt(osp.join(data_path, path.split('\\')[-1], p0_name.replace('.txt', '_plane.txt')), usecols=(0, 1, 2, -1))
        apl0 = np.zeros((plane0.shape[0], 1))
        plane0 = np.hstack((plane0, apl0))
        p0 = np.vstack((p0, plane0))
        
    p1 = np.hstack((p1, lb1, scores1))
    if tcfg.remove_plane:
        plane1 =  np.loadtxt(osp.join(data_path, path.split('\\')[-1], p1_name.replace('.txt', '_plane.txt')), usecols=(0, 1, 2, -1))
        apl1 = np.zeros((plane1.shape[0], 1))
        plane1 = np.hstack((plane1, apl1))
        p1 = np.vstack((p1, plane1))
    
    np.savetxt(osp.join(path, p0_name), p0, fmt="%.8f %.8f %.8f %.0f %.0f")
    np.savetxt(osp.join(path, p1_name), p1, fmt="%.8f %.8f %.8f %.0f %.0f")    
    
    head = '//X Y Z label scores prediction\n'
    with open(osp.join(path,p0_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p0))+'\n') + content)
    with open(osp.join(path,p1_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p1))+'\n') + content)

def save_prediction3(p0, p1, lb0, lb1, scores0, scores1, path, pc0_name, pc1_name, data_path):
    p0_name, p1_name = pc0_name[0], pc1_name[0]
    mkdir(path)
    p0 = p0[:, :, :3].squeeze(0).detach().cpu().numpy()
    p1 = p1[:, :, :3].squeeze(0).detach().cpu().numpy()
    lb0 = lb0.transpose(1,0).detach().cpu().numpy()
    lb1 = lb1.transpose(1,0).detach().cpu().numpy()
    scores0 = scores0.transpose(1,0).detach().cpu().numpy()
    scores1 = scores1.transpose(1,0).detach().cpu().numpy()
    p0 = np.hstack((p0, lb0, scores0))
    if tcfg.remove_plane:
        plane0 = np.loadtxt(osp.join(data_path, path.split('\\')[-1], p0_name.replace('.txt', '_plane.txt')), usecols=(0, 1, 2, -1))
        apl0 = np.zeros((plane0.shape[0], 1))
        plane0 = np.hstack((plane0, apl0))
        p0 = np.vstack((p0, plane0))

    p1 = np.hstack((p1, lb1*2, scores1*2))
    if tcfg.remove_plane:
        plane1 =  np.loadtxt(osp.join(data_path, path.split('\\')[-1], p1_name.replace('.txt', '_plane.txt')), usecols=(0, 1, 2, -1))
        apl1 = np.zeros((plane1.shape[0], 1))
        plane1 = np.hstack((plane1, apl1))
        p1 = np.vstack((p1, plane1))
    
    np.savetxt(osp.join(path, p0_name), p0, fmt="%.8f %.8f %.8f %.0f %.0f")
    np.savetxt(osp.join(path, p1_name), p1, fmt="%.8f %.8f %.8f %.0f %.0f")    
    
    head = '//X Y Z label scores prediction\n'
    with open(osp.join(path,p0_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p0))+'\n') + content)
    with open(osp.join(path,p1_name), 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(head + (str(len(p1))+'\n') + content)

def search_k_neighbors(raw, query, k):
    search_tree = KDTree(raw)
    _, neigh_idx = search_tree.query(query, k)
    return neigh_idx
    

def adjust_learning_rate(optimizer, epoch, lr_start=1e-4, lr_max=1e-3, lr_min=1e-6, lr_warm_up_epoch=20,
                         lr_sustain_epochs=0, lr_exp_decay=0.8):
    # warm-up strategy
    if epoch < lr_warm_up_epoch:
        lr = (lr_max - lr_start) / lr_warm_up_epoch * epoch + lr_start
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch < lr_warm_up_epoch + lr_sustain_epochs:
        lr = lr_max
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_warm_up_epoch - lr_sustain_epochs) + lr_min
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr