import os
import torch
from decimal import Decimal
import torch.nn.functional as F
import torch.nn as nn
from util.util import *
import time
import cv2
import swanlab
from torch.autograd import Variable
import matplotlib.pyplot as plt
# 
def l2_normalize(x):
    # x: numpy array (N, D)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10  # 防止除零
    return x / norm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss):
        self.args = args
        self.loader_train = loader.loader_train()
        self.loader_test = loader.loader_test()
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer(args, self.model)
        self.params_path1 = self.args.stage1_model_path
        self.params_path2 = self.args.stage2_model_path
        os.makedirs(self.params_path1, exist_ok=True)
        os.makedirs(self.params_path2, exist_ok=True)
        self.best_psnr1 = args.best_psnr
        self.best_psnr2 = args.best_psnr
        self.output_folder1 = args.output1
        self.output_folder2 = args.output2
        os.makedirs(self.output_folder1, exist_ok=True)
        os.makedirs(self.output_folder2, exist_ok=True)
        self.stage2_loaded = args.stage2_loaded
        # stage
        self.stage = 1
        # early stop
        # 学习率表
        self.lr_list = [1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6, 1e-6]
        self.lr_index = 0
        self.flag = 0  # 记录当前学习率持续epoch
        self.delete = args.delete
        #swanlab初始化
        swanlab.init(
             project=args.project_name,  # 项目名称
             config=vars(self.args),  # 自动记录所有超参数
             experiment_name="p19",  # 实验命名
           )
        self.device = next(self.model.parameters()).device

        if self.args.load == "1":
            #self.model.load_state_dict(torch.load('./params/P19/tf_time_unet_time1/39.1661.pth'), strict=False)
            self.model.load_state_dict(torch.load('./params/P19/ref_expo1/37.1296.pth'), strict=False)
            print("加载权重文件用于测试一阶段")
        if self.args.load == "2":  
            #self.model.load_state_dict(torch.load('./params/P19/tf_time_unet_time2/38.4340.pth'), strict=False) #final_6c  
            #self.model.load_state_dict(torch.load('./params/P19/tf_time_mul/35.5567.pth'), strict=False) #model2
            #self.model.load_state_dict(torch.load('./params/HIV/23.7429.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/P19/3c_2/37.5927.pth'), strict=False) #final_3c
            #self.model.load_state_dict(torch.load('./params/P19/noprior2/37.2619.pth'), strict=False)# buyong encoder gt 
            #self.model.load_state_dict(torch.load('./params/P19/notpae/38.3059.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/canon/our6c_2/37.9012.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/P19/Loss/cosine/33.5234_38.3176.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/P19/Loss/L1/33.2139_38.2957.pth'), strict=False)\
            #self.model.load_state_dict(torch.load('./params/vimeo/stage_4/34.7005_37.5401.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/challenge123/nobn_4/26.8350_38.8771.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/P19/nobn_2/33.4460_38.6858.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/P19/ref_expo2/33.7290_38.6980.pth'), strict=False)
            #self.model.load_state_dict(torch.load('./params/challenge123/ref_expo2/26.9278_38.9864.pth'), strict=False)
            
            print("加载权重文件用于测试二阶段")
      
    def train(self, epoch):
        lr = self.lr_list[self.lr_index]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.model.train()
        if self.stage == 2 and not self.stage2_loaded:
            phase1_model_path = os.path.join(self.params_path1, f'{self.best_psnr1:.4f}.pth')
            #phase1_model_path = os.path.join(self.params_path1, '36.2150_40.1006.pth')
            if os.path.exists(phase1_model_path):
                state_dict = torch.load(phase1_model_path,map_location=f'cuda:{self.args.gpu}')
                self.model.load_state_dict(state_dict, strict=False)
                self.stage2_loaded = True  # 标志置为 True，后续不再加载
                print(f"Loaded phase1 model from {phase1_model_path}")
            else:
                print("Phase1 model file not found. Please check the path!")
                self.stage2_loaded = True

        total_loss1 = 0.0
        total_loss22 = 0.0
        total_loss21 = 0.0
        total_loss2 = 0.0
        total_psnr = 0.0
        total_samples = 0
        start = time.time()
        for batch, sample in enumerate(self.loader_train):
            self.optimizer.zero_grad()
            sample = {k: v.to(self.device) for k, v in sample.items()}
            low = sample['low_image']
            mid = sample['mid_image']
            high = sample['high_image']
            gt = sample['hdr_gt']
            expos = sample['expos']
            expos_att = sample['expos_att']
            low1 = low[:, :3, :, :]
            high1 = high[:, :3, :, :]
            expos_att = expos_att.to(torch.float32)
            # forward
            # train stage1
            if self.stage == 1:
                ###mlp
                fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,m,n= self.model((sample, False))
                ###no mlp
                #fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr= self.model((sample, False))
                hdr = range_compressor_tensor(hdr)
                hdr = torch.clamp(hdr, 0., 1.)
                PSNR = batch_PSNR(hdr, gt, 1)
                total_psnr += PSNR * mid.shape[0]
                loss_low = self.loss(_low, low1,'L1')
                loss_high = self.loss(_high, high1,'L1')
                loss_hdr = self.loss(hdr, gt,'L1')
                loss1 = loss_low + loss_high + loss_hdr
                total_loss1 += loss1.item() * mid.shape[0]
                loss = loss1
            # train stage2
            else:
                fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2 = self.model((sample, True))
                #fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr= self.model((sample, True))
                #_low, _high, hdr= self.model((sample, True))
                hdr = range_compressor_tensor(hdr)
                hdr = torch.clamp(hdr, 0., 1.)
                PSNR = batch_PSNR(hdr, gt, 1)
                total_psnr += PSNR * mid.shape[0]
                loss_fea1 = self.loss(l_f2, fea1,'MSE')
                loss_fea2 = self.loss(h_f2 , fea2,'MSE')
                loss_low = self.loss(_low, low1,'L1')
                loss_high = self.loss(_high, high1,'L1')
                loss_hdr = self.loss(hdr, gt,'L1')
                loss1 = loss_low + loss_high + loss_hdr
                loss2 = loss_fea1 + loss_fea2
                #loss2 = loss1
                loss = loss2 + loss1
                #loss = loss1
                total_loss21 += loss1.item() * mid.shape[0]
                total_loss22 += loss2.item() * mid.shape[0]
                total_loss2 += loss.item() * mid.shape[0]

            # backward
            loss.backward()
            self.optimizer.step()
            total_samples += mid.shape[0]
            if self.stage == 1:
                if (batch + 1) % self.args.print_every == 0:
                    print(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [loss1: {:.4f}]\t'
                        'PSNR: {:.4f}\t'
                        'lr: {:.6f}\t'
                        'Time [{:.4f}s]'
                        'flag:[{}]'.format(
                            epoch, (batch + 1), len(self.loader_train),
                            loss1.item(),
                            PSNR,
                            lr,
                            time.time() - start,
                            self.flag
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    print(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [loss2:{:.4f}] [loss1:{:.4f}]\t'
#                        'Loss [loss2:{:.4f}]\t'
                        'PSNR: {:.4f}\t'
                        'lr: {:.6f}\t'
                        'Time [{:.4f}s]'
                        'flag:[{}]'.format(
                            epoch, (batch + 1), len(self.loader_train),
                            loss2.item(), loss1.item(),
                            PSNR,
                            lr,
                            time.time() - start,
                            self.flag
                        ))
        avg_loss1 = total_loss1 / total_samples
        avg_psnr = total_psnr / total_samples
        avg_loss21 = total_loss21 / total_samples if epoch > self.args.epochs_encoder else None
        avg_loss22 = total_loss22 / total_samples if epoch > self.args.epochs_encoder else None
        avg_loss2 = total_loss2 / total_samples if epoch > self.args.epochs_encoder else None
        swanlab.log({
            "stage1:loss": avg_loss1,
            "stage2:loss1": avg_loss21,
            "stage2:loss2": avg_loss22,
            "stage2:loss": avg_loss2,
            "PSNR": avg_psnr,
            "lr": lr,
        }, step=epoch)
        return epoch, self.stage
    def tt(self,epoch):
        print("epoch:",epoch)
        if epoch >= 50:
            self.stage = 2
        else:
            self.stage = 1
        return self.stage
    def test1_nocut(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                expos = sample['expos']
                expos_att = sample['expos_att']
                h, w = mid.shape[2], mid.shape[3]
                h = h - h % 16
                w = w - w % 16
                low = low[:, :, :h, :w]
                mid = mid[:, :, :h, :w]
                high = high[:, :, :h, :w]
                gt = gt[:, :, :h, :w]
                hdr,lo,hi = self.model((sample, False))
                PSNR_l = batch_PSNR(hdr, gt, 1)
                val_psnr_l = val_psnr_l + PSNR_l * mid.shape[0]
                hdr = range_compressor_tensor(hdr)
                gt = range_compressor_tensor(gt)
                PSNR_u = batch_PSNR(hdr, gt, 1)
                print(f"\rPSNR_u == {PSNR_u}", end='')
                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                cv2.imwrite(f'{self.output_folder1}/' + str(idx + 1) + '.tif', pre)
                val_psnr_u = val_psnr_u + PSNR_u * mid.shape[0]
            end = time.time()
            val_psnr_u = val_psnr_u / len(self.loader_test)
            val_psnr_l = val_psnr_l / len(self.loader_test)
            # #记录验证指标
            swanlab.log({
                      "val/PSNR_u": val_psnr_u,
                      "val/PSNR_l": val_psnr_l
                       }, step=epoch)
            if self.best_psnr1 < val_psnr_u:
                save_path = os.path.join(self.params_path1, f'{val_psnr_u:.4f}.pth')
                torch.save(self.model.state_dict(), save_path)
                self.best_psnr1 = val_psnr_u
                self.flag = 0
            else:
                self.flag += 1
            if self.flag >= 6:
                self.lr_index += 1
                self.flag = 0
            if self.lr_index == len(self.lr_list):
                self.lr_index = 0
                self.flag = 0
                self.stage = 2
            print(f"psnr_l= {val_psnr_l:.4f}, psnr_u= {val_psnr_u:.4f}, time= {end - start:.2f},flag= {self.flag}")
            return self.stage

    def test2_nocut(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                h, w = mid.shape[2], mid.shape[3]
                h = h - h % 16
                w = w - w % 16
                low = low[:, :, :h, :w]
                mid = mid[:, :, :h, :w]
                high = high[:, :, :h, :w]
                gt = gt[:, :, :h, :w]
                expos = sample['expos']
                expos_att = sample['expos_att']
                hdr,lo,hi = self.model((sample,True))

     #           _hdr = hdr[0].detach().cpu().numpy()
      #          _hdr = np.transpose(_hdr, (1, 2, 0))
        #        _hdr = _hdr.astype(np.float32)
       #         cv2.imwrite(f'{self.output_folder2}/' + str(idx + 1) + '.hdr', _hdr)

                PSNR_l = batch_PSNR(hdr, gt, 1)
                val_psnr_l = val_psnr_l + PSNR_l * mid.shape[0]
                hdr = range_compressor_tensor(hdr)
                gt = range_compressor_tensor(gt)
                PSNR_u = batch_PSNR(hdr, gt, 1)
                print(f"\rPSNR_u == {PSNR_u}", end='')
                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                cv2.imwrite(f'{self.output_folder2}/' + str(idx + 1) + '.tif', pre)
                val_psnr_u = val_psnr_u + PSNR_u * mid.shape[0]
            end = time.time()
            val_psnr_u = val_psnr_u / len(self.loader_test)
            val_psnr_l = val_psnr_l / len(self.loader_test)
            # 记录验证指标
            swanlab.log({
                      "val/PSNR_u": val_psnr_u,
                      "val/PSNR_l": val_psnr_l
                       }, step=epoch)
            if self.best_psnr2 < val_psnr_u:
                save_path = os.path.join(self.params_path2, f'{val_psnr_l:.4f}_{val_psnr_u:.4f}.pth')
                torch.save(self.model.state_dict(), save_path)
                self.best_psnr2 = val_psnr_u
                self.flag = 0
            else:
                self.flag += 1
            if self.flag >= 6:
                self.lr_index += 1
                self.flag = 0
            if self.lr_index == len(self.lr_list):
                self.lr_index = 0
                self.flag = 0
                self.stage = -1
            print(f"psnr_l= {val_psnr_l:.4f}, psnr_u= {val_psnr_u:.4f}, time= {end - start:.2f},stage= {self.stage})")
            return self.stage

    def test1_result(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                expos = sample['expos']
                expos_att = sample['expos_att']
                hdr,_low,_high,fea_diff1, fea_diff2, fea1, fea2 = self.model((sample, False))

                PSNR_l = batch_PSNR(hdr, gt, 1)
                val_psnr_l = val_psnr_l + PSNR_l * mid.shape[0]
                hdr = range_compressor_tensor(hdr)
                gt = range_compressor_tensor(gt)
                PSNR_u = batch_PSNR(hdr, gt, 1)
                print(f"\rPSNR_u == {PSNR_u}", end='')
                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                cv2.imwrite(f'./output/P19/ref_expo1/{idx + 1}_{PSNR_u:.2f}.tif', pre)
#                save_heatmap(m, "encoder1", f"{idx+1}_m.png")
#                save_heatmap(R1_out, "encoder1", f"{idx+1}_R1out.png")
#
#                save_heatmap(m2, "encoder2", f"{idx+1}_m2.png")
#                save_heatmap(R2_out, "encoder2", f"{idx+1}_R2out.png")
# 
#                save_heatmap(m3, "encoder3", f"{idx+1}_m3.png")
#                save_heatmap(R3_out, "encoder3", f"{idx+1}_R3out.png")
# 
#                save_heatmap(m4, "encoder4", f"{idx+1}_m4.png")
#                save_heatmap(R4_out, "encoder4", f"{idx+1}_R4out.png")
#
#                save_heatmap(m5, "encoder5", f"{idx+1}_m5.png")
#                save_heatmap(R5_out, "encoder5", f"{idx+1}_R5out.png")
                val_psnr_u = val_psnr_u + PSNR_u * mid.shape[0]
            end = time.time()
            val_psnr_u = val_psnr_u / len(self.loader_test)
            val_psnr_l = val_psnr_l / len(self.loader_test)
            print(f"psnr_l= {val_psnr_l:.4f}, psnr_u= {val_psnr_u:.4f}, time= {end - start:.2f}")



    def test2_result1(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        val_ssim_u = 0
        val_ssim_l = 0
        total_model_time = 0.0
        import pandas as pd
        fea1_all, fea2_all, fea_diff1_all, fea_diff2_all = [], [], [], []
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                expos = sample['expos']
                expos_att = sample['expos_att']
                h, w = mid.shape[2], mid.shape[3]
                h = h - h % 16
                w = w - w % 16
                low = low[:, :, :h, :w]
                mid = mid[:, :, :h, :w]
                high = high[:, :, :h, :w]
                gt = gt[:, :, :h, :w]
                low1 = low[:, :3, :, :]
                high1 = high[:, :3, :, :]

                torch.cuda.synchronize()
                model_start = time.perf_counter()
                #hdr,_low,_high = self.model((sample,True))
                hdr,_low,_high,fea_diff1, fea_diff2, fea1, fea2 = self.model((sample,True))
                torch.cuda.synchronize()
                model_end = time.perf_counter()
                model_infer_time = model_end - model_start
                total_model_time += model_infer_time
                
                #  ##t-sne  
                fea1_all.append(fea1.detach().cpu().numpy())           # shape: (B, D)
                fea2_all.append(fea2.detach().cpu().numpy())
                fea_diff1_all.append(fea_diff1.detach().cpu().numpy())
                fea_diff2_all.append(fea_diff2.detach().cpu().numpy())
                
                #save image
                pre = formulate_hdr(_low)
                pre = np.round(pre.data.cpu().numpy()).astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                #cv2.imwrite(f'{self.output_folder1}/{idx + 1}_low.tif', pre)  
                
                ppd = formulate_hdr(_high)
                ppd= np.round(ppd.data.cpu().numpy()).astype(np.uint16)
                ppd= ppd.transpose(1, 2, 0)
                #cv2.imwrite(f'{self.output_folder1}/{idx + 1}_high.tif', ppd)               
                
      
                PSNR_l,SSIM_l = batch_PSNR_SSIM(hdr, gt, 1)
                val_psnr_l = val_psnr_l + PSNR_l * mid.shape[0]
                val_ssim_l = val_ssim_l + SSIM_l * mid.shape[0]

                pre = hdr[0]               # shape [C, H, W]
                pre = pre.permute(1, 2, 0) # shape [H, W, C]
                pre = pre.data.cpu().numpy().astype(np.float32)
                #cv2.imwrite(f'{self.output_folder2}/{idx + 1}.hdr', pre)

                hdr = range_compressor_tensor(hdr)
                gt = range_compressor_tensor(gt)



                PSNR_u,SSIM_u = batch_PSNR_SSIM(hdr, gt, 1)
                print(f"\rPSNR_u == {PSNR_u}", end='')

                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                #cv2.imwrite(f'{self.output_folder2}/{idx + 1}_{PSNR_u:.4f}.tif', pre)

                val_psnr_u = val_psnr_u + PSNR_u * mid.shape[0]
                val_ssim_u = val_ssim_u + SSIM_u * mid.shape[0]
            end = time.time()

            fea1_all = np.concatenate(fea1_all, axis=0)
            fea2_all = np.concatenate(fea2_all, axis=0)
            fea_diff1_all = np.concatenate(fea_diff1_all, axis=0)
            fea_diff2_all = np.concatenate(fea_diff2_all, axis=0)

            #fea1_all_norm = l2_normalize(fea1_all)
            #fea2_all_norm = l2_normalize(fea2_all)
            #fea_diff1_all_norm = l2_normalize(fea_diff1_all)
            #fea_diff2_all_norm = l2_normalize(fea_diff2_all)
            # 保存 CSV
            pd.DataFrame(fea1_all).to_csv("fea1.csv", index=False,float_format="%.8f")
            pd.DataFrame(fea2_all).to_csv("fea2.csv", index=False,float_format="%.8f")
            pd.DataFrame(fea_diff1_all).to_csv("fea_diff1.csv", index=False,float_format="%.8f")
            pd.DataFrame(fea_diff2_all).to_csv("fea_diff2.csv", index=False,float_format="%.8f")

            val_psnr_u = val_psnr_u / len(self.loader_test)
            val_psnr_l = val_psnr_l / len(self.loader_test)
            val_ssim_u = val_ssim_u / len(self.loader_test)
            val_ssim_l = val_ssim_l / len(self.loader_test)

            avg_model_time = total_model_time / len(self.loader_test)
            print(f"Average model inference time per image: {avg_model_time:.4f} seconds")
            print(f"psnr_l= {val_psnr_l:.4f}, psnr_u= {val_psnr_u:.4f},ssim_l= {val_ssim_l:.4f}, ssim_u= {val_ssim_u:.4f}, time= {end - start:.2f}")
            #print(f"psnr_l= {val_psnr_l:.4f}, ssim_l= {val_ssim_l:.4f}, time= {end - start:.2f}")


    def inference(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        val_ssim_u = 0
        val_ssim_l = 0
        total_model_time = 0.0
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                sample = {k: v.to(self.device) for k, v in sample.items()}
                mid = sample['mid_image']
                expos = sample['expos']
                expos_att = sample['expos_att']
                h, w = mid.shape[2], mid.shape[3]
                h = h - h % 16
                w = w - w % 16
                mid = mid[:, :, :h, :w]
                torch.cuda.synchronize()
                model_start = time.perf_counter()
                hdr,_low,_high = self.model((sample,True))
                torch.cuda.synchronize()
                model_end = time.perf_counter()
                model_infer_time = model_end - model_start
                total_model_time += model_infer_time               
   
                hdr = range_compressor_tensor(hdr)
                #save image
                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                cv2.imwrite(f'{self.output_folder2}/{idx + 1}.tif', pre)
            end = time.time()
            avg_model_time = total_model_time / len(self.loader_test)
            print(f"Average model inference time per image: {avg_model_time:.4f} seconds")

    def test2_result(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        
        # 新增部分：初始化特征收集容器
        fea1_list = []
        s_fea1_list = []
        fea_diff1_list = []
        c1_list =[]
        fea2_list = []
        s_fea2_list = []
        fea_diff2_list = []
        c2_list =[]
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                # 原始数据处理流程
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                expos = sample['expos']
                expos_att = sample['expos_att']
                
                # 获取模型输出
                fea_diff1, fea_diff2, fea1, fea2, hdr= self.model((sample, True))
                
                # 新增部分：收集特征向量
                fea1_list.append(fea1.squeeze(0).cpu().numpy())      # (256,)
                #s_fea1_list.append(s_fea1.squeeze(0).cpu().numpy())  # (256,)
                fea_diff1_list.append(fea_diff1.squeeze(0).cpu().numpy())  # (256,)
                #c1_list.append(c1.squeeze(0).cpu().numpy())
                fea2_list.append(fea2.squeeze(0).cpu().numpy())      # (256,)
                #s_fea2_list.append(s_fea2.squeeze(0).cpu().numpy())  # (256,)
                fea_diff2_list.append(fea_diff2.squeeze(0).cpu().numpy())  # (256,)
                #c2_list.append(c2.squeeze(0).cpu().numpy())
    
                # 原始PSNR计算
                PSNR_l = batch_PSNR(hdr, gt, 1)
                val_psnr_l = val_psnr_l + PSNR_l * mid.shape[0]
                hdr = range_compressor_tensor(hdr)
                gt = range_compressor_tensor(gt)
                PSNR_u = batch_PSNR(hdr, gt, 1)
                val_psnr_u = val_psnr_u + PSNR_u * mid.shape[0]
                pre = formulate_hdr(hdr)
                pre = pre.data.cpu().numpy().astype(np.uint16)
                pre = pre.transpose(1, 2, 0)
                #cv2.imwrite(f'{self.output_folder2}/' + str(idx + 1) + '.tif', pre)
                
    
            # 新增部分：特征可视化
            def plot_tsne1(features, labels):
                """t-SNE可视化函数"""
                from sklearn.manifold import TSNE
                from sklearn.preprocessing import StandardScaler
                import matplotlib.pyplot as plt
    
                # 标准化
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
    
                # 运行t-SNE
                tsne = TSNE(n_components=2, 
                            perplexity=30,  # 适合约100-300个样本
                            n_iter=1000,
                            random_state=42)
                embeddings = tsne.fit_transform(scaled_features)
    
                # 可视化
                plt.figure(figsize=(10, 8))
                colors = ['blue',  'red']
                markers = ['o', 'o']
                labels_name = ['fea1', 'fea_diff1']
    
                for i in range(2):
                    mask = (labels == i)
                    plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                               c=colors[i], marker=markers[i],
                               s=50, alpha=0.7,
                               edgecolors='w' if i==0 else 'k',
                               linewidths=0.5,
                               label=labels_name[i])
    
                plt.title('t-SNE Visualization of Feature Comparison')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.xlim(-30, 30)
                plt.ylim(-25, 25)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(f'{self.output_folder2}/tsne1.png')  # 保存图片
                plt.close()
            # 新增部分：特征可视化
            def plot_tsne2(features, labels):
                """t-SNE可视化函数"""
                from sklearn.manifold import TSNE
                from sklearn.preprocessing import StandardScaler
                import matplotlib.pyplot as plt
    
                # 标准化
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
    
                # 运行t-SNE
                tsne = TSNE(n_components=2, 
                            perplexity=30,  # 适合约100-300个样本
                            n_iter=1000,
                            random_state=42)
                embeddings = tsne.fit_transform(scaled_features)
    
                # 可视化
                plt.figure(figsize=(10, 8))
                colors = ['blue', 'red']
                markers = ['o', 'o']
                labels_name = ['fea2', 'fea_diff2']
    
                for i in range(2):
                    mask = (labels == i)
                    plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                               c=colors[i], marker=markers[i],
                               s=50, alpha=0.7,
                               edgecolors='w' if i==0 else 'k',
                               linewidths=0.5,
                               label=labels_name[i])
    
                plt.title('t-SNE Visualization of Feature Comparison')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(f'{self.output_folder2}/tsne2.png')  # 保存图片
                plt.close()
    
            # 合并特征并生成标签
            all_features1 = np.concatenate([fea1_list, fea_diff1_list])
            all_features2 = np.concatenate([fea2_list, fea_diff2_list])
            labels1 = np.concatenate([
                np.zeros(len(fea1_list)),        # fea1标签为0
                #np.ones(len(s_fea1_list)),       # s_fea1标签为1
                #np.full(len(fea_diff1_list), 2),  # fea_diff1标签为2
                np.ones(len(fea_diff1_list))
                
            ])
            labels2 = np.concatenate([
                np.zeros(len(fea2_list)),        # fea1标签为0
                #np.ones(len(s_fea2_list)),       # s_fea1标签为1
                #np.full(len(fea_diff2_list), 2),  # fea_diff1标签为2
                np.ones(len(fea_diff2_list))
            ])
            
            # 执行可视化
            plot_tsne1(all_features1, labels1)
            plot_tsne2(all_features2, labels2)
            # 原始输出
            end = time.time()
            val_psnr_u = val_psnr_u / len(self.loader_test)
            val_psnr_l = val_psnr_l / len(self.loader_test)
            print(f"psnr_l= {val_psnr_l:.4f}, psnr_u= {val_psnr_u:.4f}, time= {end - start:.2f}")


    def test3_result(self, epoch):
        self.model.eval()
        start = time.time()
        val_psnr_u = 0
        val_psnr_l = 0
        
        # 新增部分：初始化特征收集容器
        fea1_distances = []
        fea2_distances = []
        s1 = []
        s2 = []
        with torch.no_grad():
            for idx, sample in enumerate(self.loader_test):
                # 原始数据处理流程
                sample = {k: v.to(self.device) for k, v in sample.items()}
                low = sample['low_image']
                mid = sample['mid_image']
                high = sample['high_image']
                gt = sample['hdr_gt']
                expos = sample['expos']
                expos_att = sample['expos_att']
                
                # 获取模型输出
                fea_diff1, fea_diff2, fea1, fea2, hdr= self.model((sample, True))
                fea1 = F.normalize(fea1, p=2, dim=1)
                fea_diff1 = F.normalize(fea_diff1, p=2, dim=1)
                fea2 = F.normalize(fea2, p=2, dim=1)
                fea_diff2 = F.normalize(fea_diff2, p=2, dim=1)
                cos_sim1 = F.cosine_similarity(fea1, fea_diff1, dim=1)
                cos_sim2 = F.cosine_similarity(fea2, fea_diff2, dim=1)
                s1.extend(cos_sim1.cpu().numpy())
                s2.extend(cos_sim2.cpu().numpy())
                
                dist1 = torch.norm(fea_diff1 - fea1, p=2, dim=1)
               
                dist2 = torch.norm(fea_diff2 - fea2, p=2, dim=1)

                fea1_distances.append(dist1.cpu())
                fea2_distances.append(dist2.cpu())
        mean_cos_sim1 = sum(s1) / len(s1)
        mean_cos_sim2 = sum(s2) / len(s2)
        print(f"平均余弦相似度: {mean_cos_sim1:.4f},{mean_cos_sim2:.4f}")
        all_fea1_dists = torch.cat(fea1_distances)
        all_fea2_dists = torch.cat(fea2_distances)
        fea1_mean = all_fea1_dists.mean().item()
        fea1_std = all_fea1_dists.std().item()
        fea2_mean = all_fea2_dists.mean().item()
        fea2_std = all_fea2_dists.std().item()
        save_path = os.path.join(self.output_folder2, f'distance.png')
        print(f"\n{'='*50}")
        print(f"特征距离分析 (Epoch {epoch}):")
        print(f"{'='*50}")
        print(f"第一对向量(fea_diff1 vs fea1):")
        print(f"  平均距离: {fea1_mean:.4f} ± {fea1_std:.4f}")
        print(f"  最小距离: {all_fea1_dists.min().item():.4f}")
        print(f"  最大距离: {all_fea1_dists.max().item():.4f}") 
        print(f"\n第二对向量(fea_diff2 vs fea2):")
        print(f"  平均距离: {fea2_mean:.4f} ± {fea2_std:.4f}")
        print(f"  最小距离: {all_fea2_dists.min().item():.4f}")
        print(f"  最大距离: {all_fea2_dists.max().item():.4f}")
        print(f"{'='*50}")
        plt.figure(figsize=(12, 6))    
        plt.subplot(1, 2, 1)
        plt.hist(all_fea1_dists.numpy(), bins=30, alpha=0.7, color='blue')
        plt.axvline(fea1_mean, color='red', linestyle='dashed', linewidth=1)
        plt.title('fea_diff1 vs fea1 distance distribution')
        plt.xlabel('L2 distance')
        plt.ylabel('number of samples')  
        plt.ylim(0, 15)
        plt.subplot(1, 2, 2)
        plt.hist(all_fea2_dists.numpy(), bins=30, alpha=0.7, color='green')
        plt.axvline(fea2_mean, color='red', linestyle='dashed', linewidth=1)
        plt.title('fea_diff2 vs fea2 distance distribution')
        plt.xlabel('L2 distance')  
        plt.ylim(0, 15)
        plt.tight_layout()
        # 高质量保存图像[8,9](@ref)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化结果已保存至: {save_path}")
