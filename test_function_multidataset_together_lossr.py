import os
import cv2
import copy
from time import time
import math
from skimage.measure import compare_ssim as ssim
import torch
import numpy as np
import glob
import scipy.io as sio
####################################################################################
class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        z_hat = torch.ifft(fftz * mask, 2)
        x = z_hat[:, :, :, 0:1]
        x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
        return x
##########################################################################################
def compute_measure(y_gt, y_pred, data_range):
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))
#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#####################################################################################
# test implement
def test_implement_MRI(test_dir, model, cs_ratio,cond, mask, test_name, epoch_num, result_dir,run_mode,loss_mode,device):
    print('\n')
    print("CS Reconstruction Start")
    filepaths = glob.glob(test_dir + '/*')
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    RMSE_total = []
    time_all =0
    with torch.no_grad():
        for img_no in range(ImgNum):
            if test_name =='BrainImages_test'or test_name == 'heart_test':
                imgName = filepaths[img_no]
                Iorg = cv2.imread(imgName, 0)
                Icol = Iorg.reshape(1, 1, 256, 256) / 255.0
            elif test_name == 'ChestTest' or test_name == 'complexMRITest':
                imgName = filepaths[img_no]
                Iorg_mat = sio.loadmat(imgName)
                Iorg = Iorg_mat['im_ori']
                Icol = Iorg.reshape(1, 1, 256, 256)

            Img_output = Icol
            start = time()
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            if loss_mode == 'ISTAplus':
                [x_output, loss_layers_sym] = model(PhiTb, mask)
            elif loss_mode == 'Fista':
                [x_output, loss_layers_sym, encoder_st] = model(PhiTb, mask)
            elif loss_mode == 'BHI':
                [x_output, x_mid] = model(PhiTb, mask)
            elif loss_mode == 'L2':
                [x_output] = model(PhiTb)
            elif loss_mode == 'Together':
                [x_output, x_mid,r_mid,r_final] = model(cond,PhiTb, mask,batch_x)
            elif loss_mode == 'Together1':
                [x_output, x_mid] = model(cond, PhiTb, mask,batch_x)
            end = time()
            time_all = time_all + (end - start)
            Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
            X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
            if test_name == 'BrainImages_test' or test_name == 'heart_test':
                rec_PSNR = psnr(X_rec * 255., Iorg.astype(np.float64))
                rec_SSIM = ssim(X_rec * 255., Iorg.astype(np.float64), data_range=255)
                m_reg = compute_measure(Iorg.astype(np.float64)/255, X_rec, 1)
            elif test_name == 'ChestTest' or test_name == 'complexMRITest':
                rec_PSNR = psnr(X_rec* 255, Iorg.astype(np.float64)* 255)
                rec_SSIM = ssim(X_rec, Iorg.astype(np.float64), data_range=1)
                m_reg = compute_measure(Iorg.astype(np.float64), X_rec, 1)
            if run_mode == 'test':
                print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.3f, Proposed SSIM is %.5f, Proposed RMSE is %.5f" % (
                img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM,m_reg))
                im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
                img_name = imgName.split('/')
                img_name = str(img_name[2])
                img_name = img_name.split('.')
                img_dir = result_dir + img_name[0] + "_PSNR_%.3f_SSIM_%.5f_RMSE_%.5f.png" % (rec_PSNR, rec_SSIM,m_reg)
                cv2.imwrite(img_dir, im_rec_rgb)
            del x_output
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            RMSE_total.append(m_reg)

    # save result
    output_file_name = result_dir + 'result_psnr_all.txt'
    output_data_psnr = [PSNR_All]
    output_file = open(output_file_name, 'a')
    for fp in output_data_psnr:  # write data in txt
        output_file.write(str(fp))
        output_file.write(',')
    output_file.write('\n')  # line feed
    output_file.close()

    print('\n')
    output_data = "CS ratio is %d, Avg time is %.5f, Avg Proposed PSNR/SSIM/RMSE for %s is %.3f-%.3f/%.5f-%.5f/%.5f-%.5f, Epoch number of model is %d \n" % (
        cs_ratio, time_all/ImgNum,test_name, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All),
        np.array(RMSE_total).mean(), np.array(RMSE_total).std(), epoch_num)
    print(output_data)
    print("CS Reconstruction End")
    return np.mean(PSNR_All), np.mean(SSIM_All),np.array(RMSE_total).mean()
##############################################################################################################