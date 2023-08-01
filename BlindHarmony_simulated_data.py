import os
import numpy as np
import scipy.misc
import sacred
import torch
from tqdm import tqdm
import sys
import natsort, glob, pickle, torch
import numpy as np
import os
import torch.distributions
import scipy
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import nde.flows, nde.transforms
import json
from flow_model import  create_flow
class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
def pickleRead(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def find_files(wildcard): return natsort.natsorted(glob.glob(wildcard, recursive=True))
def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=1).astype(np.float32)).cuda()
def rgb(t): return ((t[:,0,:,:] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]))

def load_pkls( path,nmax = -1,nmin = 0):
    assert os.path.isfile(path), path
    images = []
    with open(path, "rb") as f:
        images += pickle.load(f)
    assert len(images) > 0, path
    if nmin > 0 or nmax > 0:
        images = images[nmin:nmax]
    else:
        images = images[:]
    return images

# Preprocessing
def dequant(img, num_bits=12):
    """
    input: image with either int8 or float (range [0,1]) pixel values
    output: uniformly dequantized image with float (range [0,2**num_bits-1]) pixel values
    """   
    
    if img.dtype == torch.uint8:
        img = img.float() # Already in [0,255]
    else:
        img = img * 255. # [0,1] -> [0,255]

    if num_bits != 8:
        img = torch.floor(img / 2 ** (8 - num_bits)) # [0, 255] -> [0, num_bins - 1]

    # Uniform dequantization.
    img = img + torch.rand_like(img)

    return img

def dequant_inverse(inputs, num_bits=12):
    """
    input: uniformly dequantized image with float (range [0,2**num_bits-1]) pixel values
    output: image with float (range [0,1]) pixel values
    """
    # Discretize the pixel values.
    
    inputs = torch.floor(inputs)
    # Convert to a float in [0, 1].
    inputs = inputs * (256 / 2**num_bits) / 255
    inputs = torch.clamp(inputs, 0, 1)
    return inputs

def min_max(img):
    """
    input: image with float32 pixel values
    output: min-max normalized image with float32 (range [0,1]) pixel values
    """
    img_min, img_max = img.min(), img.max()
    img_normalized = ((img - img_min)/(img_max - img_min)).astype('float32')
    return torch.from_numpy(img_normalized)

def sobel(image):
    dx = cv2.Sobel(image,cv2.CV_64F,1,0,3)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1,3)
    lr = np.expand_dims(cv2.magnitude(dx,dy),2)
    return lr, np.abs(dx), np.abs(dy)

def BlindHarmony(img,model,alpha,beta_fidel,beta_grad,num,init,gt):
    with torch.no_grad():

        _,dx,dy = sobel(img)
        Gmx = (dx<np.percentile(dx,30,axis=(0,1), keepdims=True))*(img >.05)
        Gmy = (dy<np.percentile(dy,30,axis=(0,1), keepdims=True))*(img >.05)
        sr_0 = init
        z, logabsdet = model._transform(dequant(t(sr_0)), context=None)

        sr = sr_0
        for temperature in range(0, num):
            dx = Gmx*cv2.Sobel(sr,cv2.CV_64F,1,0,3)
            dy = Gmy*cv2.Sobel(sr,cv2.CV_64F,0,1,3)
            grad_grad= -(cv2.Sobel(np.sign(dx),cv2.CV_64F,1,0,3) + cv2.Sobel(np.sign(dy),cv2.CV_64F,0,1,3))
            grad_fidel = -img/(1e-6+np.sqrt(np.sum(sr**2,(0,1), keepdims=True)))/(1e-6+np.sqrt(np.sum(img**2,(0,1), keepdims=True))) \
                + np.sum(sr*img,(0,1), keepdims=True)*sr/(1e-6+np.sqrt(np.sum(sr**2,(0,1), keepdims=True))**3)/(1e-6+np.sqrt(np.sum(img**2,(0,1), keepdims=True)))
            
            grad = beta_fidel*grad_fidel + beta_grad*grad_grad
            
            sr_grad = sr - grad
            z, logabsdet = model._transform(dequant(t(sr_grad)), context=None)
            
            z = (1-alpha) * z
            sr, logabsdet0 = model._transform.inverse(z, context=None)
            sr = rgb(dequant_inverse(sr))
            
            if np.any(np.isnan(sr)) or np.any(np.isinf(sr)) or np.sum(sr) == 0 or np.any((sr>100)): 
                sr = sr_grad 
            
    return sr

def gen_sc(pckl,ze,batch,mode = 'exp'):
    gt = np.zeros((pckl[0].shape[0],pckl[0].shape[1],batch))
    for i in range(batch):
        tmp = (pckl[i+batch*ze])
        gt[:,:,i:i+1] = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
    if mode == 'exp':
        img_sc = np.exp(np.copy(gt))
    elif mode == 'log':
        img_sc = np.log(1+3*np.copy(gt))
    elif mode == 'gamma09':
        img_sc = (np.copy(gt))**0.9
    elif mode == 'gamma08':
        img_sc = (np.copy(gt))**0.8
    elif mode == 'gamma07':
        img_sc = (np.copy(gt))**0.7
    elif mode == 'gamma12':
        img_sc = (np.copy(gt))**1.2
    elif mode == 'gamma13':
        img_sc = (np.copy(gt))**1.3
    elif mode == 'gamma15':
        img_sc = (np.copy(gt))**1.5
    
    for i in range(batch):
        tmp = img_sc[:,:,i:i+1]
        img_sc[:,:,i:i+1] = (tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))
    return img_sc,gt

def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)
    
def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))


os.environ["CUDA_VISIBLE_DEVICES"]='1'
c,h,w = 1,144,176
run_dir = '/home/hwihun/blindharmony/nsf/runs/images/ADNI-12bit_batch256'
flow_checkpoint = run_dir + '/flow_best.pt'
conf_path = run_dir + '/config.json'

with open(conf_path) as f:
    conf_dict = json.load(f)
    config = Dict2Class(conf_dict)

device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model = create_flow(c, h, w,flow_checkpoint, config).to(device)
model.eval()

pckl = load_pkls('/fast_storage/hwihun/pkls/folder35177_resample_val.pklv4',10000,0)
pckl_tr = load_pkls('/fast_storage/hwihun/pkls/folder35177_resample_train.pklv4',10000,0)
ref = np.zeros([144, 176, 50])
for i in range(0,10000):
    slice = pckl_tr[i]
    ref[:,:,i%50:i%50+1] += (slice-np.min(slice))/(np.max(slice)-np.min(slice))/200

num = 10
init = ref

alpha = 0.001
beta_fidel = 1000.0
beta_grad = 0.001

mode = 'exp'

imgs_har = []
for ze in tqdm(range(int(len(pckl)/batch))):
    img_sc,gt = gen_sc(pckl,ze,batch,mode)
    
    init = ref
    img_BlindHarmony = BlindHarmony(img_sc,model,alpha,beta_fidel,beta_grad,num,init,gt)
    for i in range(batch):
        imgs_har.append(img_BlindHarmony[:,:,i])

to_pklv4(imgs_har, f'./inf_simul_data/harmonized_pro_{mode}_alpha1_{alpha}_fidel_{beta_fidel}_grad_{beta_grad}_num_{num}.pklv4', vebose=True)
