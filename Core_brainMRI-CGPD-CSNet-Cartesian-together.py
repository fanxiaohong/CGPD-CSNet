import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import platform
import glob
from argparse import ArgumentParser
import random
from test_function_multidataset_together_lossr import *
from torch.optim.lr_scheduler import MultiStepLR
###########################################################################################
# parameter
parser = ArgumentParser(description='brainMRI-CGPD-CSNet-Cartesian-together')
parser.add_argument('--net_name', type=str, default='brainMRI-CGPD-CSNet-Cartesian-together', help='name of net')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=13, help='D,11')
parser.add_argument('--growth-rate', type=int, default=32, help='G,32')
parser.add_argument('--num-layers', type=int, default=8, help='C,8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=15,
                    help='{Cartesian/pseudo-radial:10,20,30,40,50}{2D-random:5,10,20,30,40}')
parser.add_argument('--matrix_dir', type=str, default='Cartesian',
                    help='sampling matrix directory, pseudo-radial/2D-random/Cartesian/Cartesian_untrained')
parser.add_argument('--model_dir', type=str, default='model_MRI', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log_MRI', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='BrainImages_test',
                    help='name of test set, BrainImages_test, heart_test')
parser.add_argument('--run_mode', type=str, default='test', help='train、test')
parser.add_argument('--loss_mode', type=str, default='Together1', help='BHI,Fista，ISTAplus,L2,Together')
parser.add_argument('--save_interval', type=int, default=50, help='epoch number of each save interval')
parser.add_argument('--print_flag', type=int, default=1, help='print parameter number 1 or 0')
args = parser.parse_args()
#########################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
###########################################################################################
# parameter
batch_size = 1
Training_data_Name = 'Training_BrainImages_256x256_100.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']
nrtrain = Training_labels.shape[0]   # number of training image
print('Train data shape=',Training_labels.shape)

# define save dir
model_dir = "./%s/%s_layer_%d_denselayer_%d_lr_%f" % (args.model_dir, args.net_name,args.layer_num, args.num_layers, args.learning_rate)
test_dir = os.path.join(args.data_dir, args.test_name)   # test image dir
output_file_name = "./%s/%s_layer_%d_denselayer_%d_lr_%f.txt" % (args.log_dir, args.net_name, args.layer_num, args.num_layers, args.learning_rate)
#########################################################################################
# Load CS Sampling Matrix: phi
rand_num = 1
if args.matrix_dir == '2D-random':
    train_cs_ratio_set = [5, 10, 20, 30, 40]
elif args.matrix_dir =='Cartesian_untrained':
    train_cs_ratio_set = [15, 25, 35, 45]
else:
    train_cs_ratio_set = [10, 20, 30, 40, 50]

Phi_all = {}
Phi = {}
for cs_ratio in train_cs_ratio_set:
    Phi_data_Name = './sampling_matrix/%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
    Phi_data = sio.loadmat(Phi_data_Name)
    mask_matrix = Phi_data['mask_matrix']
    Phi_all[cs_ratio] = np.zeros((int(rand_num * 1), mask_matrix.shape[0], mask_matrix.shape[1]))

    for k in range(rand_num):
        Phi_all[cs_ratio][k, :, :] = mask_matrix[:, :]

    Phi[cs_ratio] = torch.from_numpy(Phi_all[cs_ratio]).type(torch.FloatTensor)

##########################################################################################
class ResidualBlock_basic(nn.Module):
    def __init__(self, nf):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_residual = self.relu(self.conv1(x))
        x_residual = self.conv2(x_residual)
        return x + x_residual
###########################################################################
class SmootherConv(nn.Module):
    def __init__(self, in_ch, out_ch, channel_G):
        super(SmootherConv, self).__init__()
        self.head_conv = nn.Conv2d(in_ch, channel_G, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(channel_G),
            ResidualBlock_basic(channel_G),
            ResidualBlock_basic(channel_G),
            ResidualBlock_basic(channel_G)
        )
        self.tail_conv = nn.Conv2d(channel_G, out_ch, 3, 1, 1, bias=True)

    def forward(self, input):
        x_residual = self.head_conv(input)
        x_residual = self.ResidualBlocks(x_residual)
        x_residual = self.tail_conv(x_residual)
        x_pred = input + x_residual

        return x_pred
###########################################################################
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)
###########################################################################
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, 1, kernel_size=1) # output 1 channel

    def forward(self, x):
        return x[:,0:1,:,:] + self.lff(self.layers(x))  # local residual learning
###########################################################################
class condition_network(nn.Module):
    def __init__(self,LayerNo):
        super(condition_network, self).__init__()

        self.fc1 = nn.Linear(1, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, LayerNo+LayerNo, bias=True)

        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x=x[:,0:1]

        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        num=x.shape[1]
        num=int(num/2)
        return x[0,0:num],x[0,num:]
###########################################################################
def get_cond(cs_ratio, sigma, type):
    para_cs = None
    para_noise = sigma / 5.0
    if type == 'org':
        para_cs = cs_ratio * 2.0 / 100.0
    elif type == 'org_ratio':
        para_cs = cs_ratio / 100.0

    para_cs_np = np.array([para_cs])

    para_cs = torch.from_numpy(para_cs_np).type(torch.FloatTensor)

    para_cs = para_cs.cuda()

    para_noise_np = np.array([para_noise])
    para_noise = torch.from_numpy(para_noise_np).type(torch.FloatTensor)

    para_noise = para_noise.cuda()
    para_cs = para_cs.view(1, 1)
    para_noise = para_noise.view(1, 1)
    para = torch.cat((para_cs, para_noise), 1)

    return para
###########################################################################
class FFT_image(torch.nn.Module):
    def __init__(self):
        super(FFT_image, self).__init__()

    def forward(self, x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        x_fft = fftz * mask
        x_fft = x_fft.view(x_dim_0, 2*x_dim_1, x_dim_2, x_dim_3)
        return x_fft
###########################################################################
class iFFT_image(torch.nn.Module):
    def __init__(self):
        super(iFFT_image, self).__init__()

    def forward(self, x):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 2)
        z_hat = torch.ifft(x, 2)
        x_ifft = z_hat[:, :, :, 0:1]
        x_ifft = x_ifft.view(x_dim_0, 1, x_dim_2, x_dim_3)
        return x_ifft
###########################################################################
# Define MRI-RDB Block
class BasicBlock(torch.nn.Module):
    def __init__(self,growth_rate, num_layers):
        super(BasicBlock, self).__init__()

        self.Sp = nn.Softplus()
        self.G = growth_rate
        self.C = num_layers

        self.rdb = RDB(3, self.G, self.C)  # local residual learning
        self.smoother_conv = SmootherConv(2,2, self.G)  # local residual learning

        self.conv_down = nn.Conv2d(2, 2, kernel_size=2, stride=2, padding=0)  # down sampling
        self.conv_up = nn.ConvTranspose2d(1, 1, 2, stride=2)  # up sampling

        self.FFT = FFT_image()
        self.iFFT = iFFT_image()

    def forward(self, x, fft_forback, PhiTb, mask, lambda_step,x_step,i_layer,LayerN,batch_x):
        m = x - lambda_step * fft_forback(x, mask) + lambda_step * PhiTb
        r = self.FFT(batch_x,mask) - self.FFT(m,mask)
        r_down = self.conv_down(r)
        error_down = self.smoother_conv(r_down)  # smoothing
        error_down_image = self.iFFT(error_down)
        error_up = self.conv_up(error_down_image)

        sigma = x_step.repeat(m.shape[0], 1, m.shape[2], m.shape[3])
        x_new = torch.cat((m, error_up), 1)

        x_input_cat = torch.cat((x_new, sigma), 1)
        x_pred = self.rdb(x_input_cat)  # local residual learning

        return [x_pred]
#####################################################################################################
# Define Deep Geometric Distillation Network
class DGDN(torch.nn.Module):
    def __init__(self, LayerNo, growth_rate, num_layers):
        super(DGDN, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()
        self.G = growth_rate
        self.C = num_layers

        for i in range(LayerNo):
            onelayer.append(BasicBlock(self.G, self.C))  # share feature extrator

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network(LayerNo)

    def forward(self, cond,PhiTb, mask,batch_x):

        x = PhiTb
        lambda_step,x_step = self.condition(cond)

        for i in range(self.LayerNo):
            [x] = self.fcs[i](x, self.fft_forback, PhiTb, mask, lambda_step[i],x_step[i],i,self.LayerNo,batch_x)
            if i==((self.LayerNo-1)/2):
                x_mid = x
        x_final = x

        return [x_final,x_mid]
##################################################################################3
# initial test file
result_dir = os.path.join(args.result_dir, args.test_name)
result_dir = result_dir+'_'+args.net_name+'_ratio_'+ str(args.cs_ratio)+'_epoch_'+str(args.end_epoch)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
###################################################################################
# model
model = DGDN(args.layer_num, args.growth_rate, args.num_layers)
model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
###################################################################################
if args.print_flag:  # print networks parameter number
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
####################################################################################
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
#####################################################################################
if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)
#######################################################################################
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if args.start_epoch > 0:   # train stop and restart
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, args.start_epoch)))
#########################################################################################
if args.run_mode == 'train':
    # Training loop
    for epoch_i in range(args.start_epoch+1, args.end_epoch+1):
        model = model.train()
        step = 0
        for data in rand_loader:
            step = step+1
            batch_x = data
            batch_x = torch.unsqueeze(batch_x, 1).cpu().data.numpy()
            batch_x = torch.from_numpy(batch_x).to(device)

            rand_Phi_index = np.random.randint(rand_num * 1)
            rand_cs_ratio = np.random.choice(train_cs_ratio_set)
            mask = Phi[rand_cs_ratio][rand_Phi_index]
            mask = torch.unsqueeze(mask, 2)
            mask = torch.cat([mask, mask], 2)
            mask = mask.to(device)

            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            cond = get_cond(rand_cs_ratio, 0.0, 'org_ratio')

            [x_output, x_mid] = model(cond,PhiTb, mask,batch_x)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))
            loss_discrepancy_mid = torch.mean(torch.abs(x_mid - batch_x))

            loss_all = loss_discrepancy + loss_discrepancy_mid

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # step %100==0
            if step % 100 == 0:
                output_data = "[%02d/%02d] Step:%.0f | CS ratio:%.0f | Total Loss: %.6f | Discrepancy Loss: %.6f| Discrepancy mid Loss: %.6f" % \
                              (epoch_i, args.end_epoch, step,rand_cs_ratio, loss_all.item(), loss_discrepancy.item(),
                               loss_discrepancy_mid.item())
                print(output_data)

            # Load pre-trained model with epoch number
        model = model.eval()
        PSNR_mean, SSIM_mean, RMSE_mean = test_implement_MRI(test_dir, model, rand_cs_ratio,cond, mask, args.test_name,
                                                             args.end_epoch, result_dir, args.run_mode, args.loss_mode,
                                                             device)
        # save result
        output_data = [epoch_i, rand_cs_ratio, loss_all.item(), loss_discrepancy.item(), loss_discrepancy_mid.item(), PSNR_mean,
                       SSIM_mean, RMSE_mean]
        output_file = open(output_file_name, 'a')
        for fp in output_data:   # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')    # line feed
        output_file.close()

        # save model in every epoch
        if epoch_i % args.save_interval ==0:
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

elif args.run_mode=='test':
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, args.end_epoch)))
    # Load pre-trained model with epoch number
    model = model.eval()
    rand_Phi_index = np.random.randint(rand_num * 1)
    mask = Phi[args.cs_ratio][rand_Phi_index]
    mask = torch.unsqueeze(mask, 2)
    mask = torch.cat([mask, mask], 2)
    mask = mask.to(device)
    cond = get_cond(args.cs_ratio, 0.0, 'org_ratio')
    PSNR_mean, SSIM_mean, RMSE_mean = test_implement_MRI(test_dir, model, args.cs_ratio,cond, mask, args.test_name,
                                                         args.end_epoch, result_dir, args.run_mode, args.loss_mode,
                                                         device)

#########################################################################################
