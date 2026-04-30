import argparse

# import template
parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--expo_ablation_mode', type=str, default='none', choices=['none', 'first', 'third', 'both'])
parser.add_argument('--csv_log_path', type=str, default='csv/exposure_ablation_p19third.csv')


parser.add_argument('--gpu', type=int, default=0, choices=[0,1,2,3,4],
                        help='GPU id to use (default: 0)')
#video
parser.add_argument('--root_dir', type=str, default='../vimeo_septuplet',help='Path to the train dataset')

# Data specifications   # args for saving
#challenge123
#parser.add_argument('--train_path', type=str, default='data/challenge123/123train_h5/',help='Path to the train dataset')
#parser.add_argument('--test_path', type=str, default='data/challenge123/123test_h5/',help='Path to the test dataset')
#p19
parser.add_argument('--train_path', type=str, default='../xiangliang/train_h5_128/',help='Path to the train dataset')
parser.add_argument('--test_path', type=str, default='../Unet/Unet/test_h5/',help='Path to the test dataset')

#parser.add_argument('--train_path', type=str, default='../xiangliang/canon_train_h5/',help='Path to the train dataset')
#parser.add_argument('--test_path', type=str, default='../xiangliang/canon_val/',help='Path to the test dataset')

#parser.add_argument('--train_path', type=str, default='../xiangliang/k_train_h5/',help='Path to the train dataset')
#parser.add_argument('--test_path', type=str, default='../xiangliang/k_test_h5/',help='Path to the test dataset')


parser.add_argument('--stage1_model_path', type=str, default='./params/challenge123/ref_expo1/',
                    help='Path to the pth1')
parser.add_argument('--stage2_model_path', type=str, default='./params/challenge123/ref_expo2/',
                    help='Path to the pth2')
parser.add_argument('--output1', type=str, default='./output/canon/fault_expos_p19_1third',
                    help='Path to the output1')
parser.add_argument('--output2', type=str, default='./output/canon/fault_expos_p19_2third',
                    help='Path to the output2')
parser.add_argument('--print_every', type=int, default=50,
                    help='how many batches to wait before logging training status')

parser.add_argument('--stage2_loaded', default=False,
                    help='load stage1 pth or not')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')
parser.add_argument('--en_gt', type=int, default=6,
                    help='en_gt')
parser.add_argument('--en_x', type=int, default=3,
                    help='en_gt')

# Model specifications
parser.add_argument('--project_name', type=str, default="challenge123_ref", help="项目名称")
parser.add_argument('--model', default='hdr_net',
                    help='model name')
parser.add_argument('--pre_train', type=str, default= '.',
                    help='pre-trained model directory')


# Training specifications
parser.add_argument('--epochs_encoder', type=int, default=40,
                    help='number of epochs to train the degradation encoder')
parser.add_argument('--epochs_sr', type=int, default=50,
                    help='number of epochs to train the whole network')

parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--best_psnr', type=float, default=float('-inf'),
                    help='Initial best PSNR value')
parser.add_argument('--delete', default=False,
                    help='delete weight or not')
parser.add_argument('--nfeat', type=int, default=32,
                    help='number of hidden units')
parser.add_argument('--train', default=True,
                    help='train or test')
parser.add_argument('--norm', type=str, default="tif", help="图片格式")
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# diffusion specifications
parser.add_argument('--beta_schedule', type=str, default='linear',
                    help='beta_schedule')
parser.add_argument('--beta_start', type=float, default=1e-4,
                    help='beta_start')
parser.add_argument('--beta_end', type=float, default=0.02,
                    help='beta_end')
parser.add_argument('--num_diffusion_timesteps', type=int, default=500,
                    help='number of diffusion timesteps')

# inference specifications
parser.add_argument('--load', type=str, default="0",
                      help='test stage one or stage two')

# Optimization specifications
parser.add_argument('--lr_encoder', type=float, default=1e-4,
                    help='learning rate to train the degradation encoder')
parser.add_argument('--lr_sr', type=float, default=1e-4,
                    help='learning rate to train the whole network')
parser.add_argument('--lr_decay_encoder', type=int, default=8,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay_sr', type=int, default=10,
                    help='learning rate decay per N epochs')
                    
parser.add_argument('--decay_type', type=str, default='cosine',
                    help='learning rate decay type')
                    
parser.add_argument('--gamma_encoder', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
                    
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='初始学习率')
parser.add_argument('--lr_min', type=float, default=1e-6,
                    help='最小学习率')                   
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=50,
                    help='最大epoch数')
parser.add_argument('--patience', type=int, default=10,
                    help='early stop patien')
parser.add_argument('--resume', type=int, default=1200,
                    help='resume from specific checkpoint')

args = parser.parse_args()
