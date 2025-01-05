import torch.optim
from torch.utils.data import DataLoader
from torch import nn
from dataset import *
from tqdm import tqdm
from utils.utils import *
from utils.loss import MAELoss,InfoNCE
from model.pwvdswin_mae_attention import SwinMAE_attention
import argparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")

choose=True
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False
if choose==True:
    parser = argparse.ArgumentParser(description='Train TransNet')
    # parser.add_argument("--datapath", type=str, default='./20classes_8.28/mix_123.mat')
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--lamba", default=0.99)
    parser.add_argument("--alpha", default=0.2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--classesnum", type=int, default=10)
    parser.add_argument("--task", type=str, default='pretraining')  # pretraining classification
    parser.add_argument("--model_path", type=str, default=
    './checkpoint_SwinTransformer_RMLtotal_attention/pwvd/patch_size_4_window_size_2/pwvd_best_network_loss_0.04662672211007087.pth')
    parser.add_argument("--netdepth", type=int, default=64)
    parser.add_argument("--cutmixsize", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--wait", type=int, default=5)
    parser.add_argument("--declay", default=0.5)
    parser.add_argument("--yuzhi", type=int, default=10)
    parser.add_argument("--numworks", type=int, default=4)
    parser.add_argument("--pref", type=int, default=20)
    parser.add_argument("--trans_choose", type=str, default='pwvd')
    # parser.add_argument("--dataset", type=str, default='adsb')
    # parser.add_argument("--name", type=str, default='adsb')
    parser.add_argument("--dataset", type=str, default='RML2016_10a')
    parser.add_argument("--name", type=str, default='RML2016_10a')
    parser.add_argument("--withoutis", type=str, default='no')
    parser.add_argument("--RGB_is", type=str2bool, default=False)
    parser.add_argument("--adsbis", type=str2bool, default=False)
    parser.add_argument("--resample", type=str2bool, default=False)
    parser.add_argument("--chazhi", type=str2bool, default=False)
    parser.add_argument("--newdata", type=str2bool, default=False)
    parser.add_argument("--cnum", type=int, default=2)
    parser.add_argument("--samplenum", type=int, default=6)  # stft 4 pwvd 6
    opt = parser.parse_args()

def acc_classes(pre, labels,BATCH_SIZE):
    pre_y = torch.max(pre, dim=1)[1]
    train_acc = torch.eq(pre_y, labels.to(device)).sum().item() / BATCH_SIZE
    return train_acc

def acc_AA(pre, labels,acc_AA_pre,acc_AA_count):
    pre_y=torch.max(pre, dim=1)[1]
    pre_y = pre_y.detach().cpu().numpy()
    labelclass=np.array(labels)
    # labelclass[labelclass == 99] = 7
    for i in range(len(labelclass)):
        if pre_y[i]==labelclass[i]:
            acc_AA_pre[0,labelclass[i]]+=1
            acc_AA_count[0,labelclass[i]]+=1
        else:
            acc_AA_count[0, labelclass[i]] += 1
    return acc_AA_pre,acc_AA_count

def acc_snrs(pre, labels,snr,acc_snr_pre,acc_snr_count):
    pre_y = torch.max(pre, dim=1)[1]
    pre_y =pre_y.detach().cpu().numpy()
    labelclass=np.array(labels)
    for i in range(len(labelclass)):
        if pre_y[i]==labelclass[i]:
            acc_snr_pre[0,snr[i]]+=1
            acc_snr_count[0,snr[i]]+=1
        else:
            acc_snr_count[0, snr[i]] += 1
    return acc_snr_pre,acc_snr_count

def patchify(imgs,patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    if args.RGB_is==False:
        in_channel=1
    else:
        in_channel = 3
    x = imgs.reshape(shape=(imgs.shape[0], in_channel, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, p ** 2 * in_channel)
    return x

def extract_tensor(latent_noise):
    N, D = latent_noise.size()
    latent_noise_neg = torch.zeros(N, N-1, D).to(latent_noise.device)
    for i in range(N):
        latent_noise_neg[i] = torch.cat([latent_noise[:i], latent_noise[i+1:]], dim=0)

    return latent_noise_neg

def trainhec(train_loader, model, criterion,criterion2,criterion3,criterion4, optimizer, epoch, epoch_max,batchsize,adsbis=False):
    """Train for one epoch on the training set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis==True:
        acc_snr_pre=np.zeros((1,7))
        acc_snr_count = np.zeros((1,7))
    else:
        acc_snr_pre = np.zeros((1, 20))
        acc_snr_count = np.zeros((1, 20))
    # switch to train mode
    model.train()

    with tqdm(total=len(train_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3) as pbar:
        for i, (input1,input2) in enumerate(train_loader):
            images_clean, images_noise = input1, input2,

            latent_noise, latent_clean, pred_noise,pred_clean, mask= model(images_noise.to(device),images_clean.to(device))

            target_var=images_clean.to(device)
            target_var=patchify(target_var, patch_size=4)
            # target_var = target_var.to(torch.float)
            loss1 = criterion(target_var, pred_noise, mask)
            loss2 = criterion2(target_var, pred_clean, mask)
            loss3=criterion3(latent_clean,latent_noise)

            loss=args.lamba*(loss1+loss2)+(1-args.lamba)*loss3

            # measure accuracy and record loss
            acc.update(0)

            losses_class.update(loss.item())
            losses_class1.update( loss1.item())
            losses_class2.update( loss2.item())


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'train_loss_': losses_class.avg,
                                'acc': acc.avg})
            pbar.update(1)
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)

    return acc.avg, losses_class.avg


def validatehec(val_loader, model, criterion,criterion2,criterion3,criterion4, epoch, epoch_max,batchsize,adsbis=False):
    """Perform validation on the validation set"""
    losses_class = AverageMeter()
    losses_class1 = AverageMeter()
    losses_class2 = AverageMeter()
    losses_class3 = AverageMeter()
    acc = AverageMeter()
    if adsbis == True:
        acc_snr_pre_val = np.zeros((1, 7))
        acc_snr_count_val = np.zeros((1, 7))
    else:
        acc_snr_pre_val = np.zeros((1, 20))
        acc_snr_count_val = np.zeros((1, 20))
    acc_AA_pre = np.zeros((1, 11))
    acc_AA_count = np.zeros((1, 11))
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f'Epoch{epoch}/{epoch_max}', postfix=dict, mininterval=0.3,
                  colour='blue') as pbar:
            for i, (input1,input2) in enumerate(val_loader):
                images_clean, images_noise = input1, input2,

                latent_noise, latent_clean, pred_noise,pred_clean, mask= model(images_noise.to(device),images_clean.to(device))

                target_var = images_clean.to(device)
                target_var = patchify(target_var, patch_size=4)
                # target_var = target_var.to(torch.float)
                loss1 = criterion(target_var, pred_noise, mask)
                loss2 = criterion2(target_var, pred_clean, mask)
                loss3 = criterion3(latent_clean, latent_noise)

                # loss1,loss2,loss_last=1e3*loss1,1e3*loss2,1e3*loss_last
                loss = args.lamba * (loss1 + loss2) + (1 - args.lamba) * loss3

                # measure accuracy and record loss
                acc.update(0)
                # acc_AA(output, val_label,acc_AA_pre,acc_AA_count)
                losses_class.update(loss.item())
                losses_class1.update(loss1.item())
                losses_class2.update(loss2.item())


                pbar.set_postfix(**{'val_loss_class': losses_class.avg,
                                    'acc': acc.avg})
                pbar.update(1)
    print(acc_snr_pre_val / acc_snr_count_val * 100)
    print(acc_AA_pre / acc_AA_count * 100)
    print(losses_class1.avg)
    print(losses_class2.avg)
    print(losses_class3.avg)
    return acc.avg, losses_class.avg


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.trans_choose == "stft":
        train_set = SigDataSet_stft('./data/adsb_test_data.mat'.format(args.dataset), newdata=args.newdata,
                                    adsbis=args.adsbis,
                                    resample_is=args.resample, samplenum=args.samplenum, resize_is=True,
                                    norm='maxmin',
                                    snr_range=[10, 40], sgnaug=True)
    elif args.trans_choose == "pwvd":
        train_data_path = './data/{}_gaodbdata.mat'.format(args.name)
        train_set = SigDataSet_pwvd(train_data_path,newdata=args.newdata,adsbis=args.adsbis,
                                    resample_is=args.resample,samplenum=args.samplenum,resize_is=True,norm='maxmin',
                                    snr_range=[16,30],sgn_expend=True,sgnaug=True,
                                    RGB_is=args.RGB_is,zhenshiSNR=False,freq_fliter=False)
    elif args.trans_choose == "wave":
        train_set = SigDataSet_wave('./data/RML2016_10a_gaodbdata.mat'.format(args.dataset),newdata=args.newdata,adsbis=args.adsbis,
                                    resample_is=args.resample,samplenum=args.samplenum,resize_is=True,norm='maxmin',
                                    snr_range=[10,40],sgnaug=True)
    elif args.trans_choose == "gasf":
        train_set = SigDataSet_gasf('./data/RML2016.10a_total_data.mat'.format(args.dataset),newdata=args.newdata,adsbis=args.adsbis,
                                    resample_is=args.resample,samplenum=args.samplenum,resize_is=False,norm='maxmin',
                                    snr_range=[10,30],RGB_is=args.RGB_is)


    valsplit = 0.7
    train_set, val_set = torch.utils.data.random_split(train_set, [int(len(train_set) * valsplit),
                                                                   int(len(train_set)) - int(
                                                                       len(train_set) * valsplit)])

    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=args.numworks
                              , prefetch_factor=args.pref)

    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=args.numworks
                            , prefetch_factor=args.pref, shuffle=True)

    patch_size = 4
    window_size = 2
    print(args.RGB_is)
    if args.RGB_is==False:
        in_channel=1
    else:
        in_channel = 3
    model = SwinMAE_attention(
    img_size=128, patch_size=patch_size, in_chans=in_channel,
    decoder_embed_dim=384,
    depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
    window_size=window_size, qkv_bias=True, mlp_ratio=4,
    drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),mask_ratio=0.75,mask_type='suiji',len_attn=[192,256],task=args.task)


    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.cuda()

    # state_dict = torch.load(args.model_path)
    # model.load_state_dict(state_dict)


    criterion = MAELoss(norm_pix_loss=False, is_mae='smoothl1', beta=0.01, no_mask=True)
    criterion2 = MAELoss(norm_pix_loss=False, is_mae='smoothl1', beta=0.01, no_mask=False)
    criterion3=InfoNCE()
    criterion4 = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    csv_logger = CSVStats()
    early_stopping = EarlyStopping(save_path='./checkpoint_SwinTransformer_{}_attention/{}/patch_size_{}_window_size_{}'.
                                       format(args.dataset,args.trans_choose,patch_size,window_size), patience=args.patience,
                                       wait=args.wait, choose=args.trans_choose
                                    )
    wait_idem = args.wait
    declay_count = 0
    acc_train=0
    loss_train=0
    for epoch in range(0, args.epochs):

        acc_train, loss_train = trainhec(
            train_loader, model, criterion,criterion2,criterion3,criterion4, optimizer, epoch, batchsize=args.batchsize, epoch_max=args.epochs,adsbis=args.adsbis)

        torch.cuda.empty_cache()

        acc_val, loss_val = validatehec(
            val_loader, model, criterion,criterion2,criterion3,criterion4, epoch, batchsize=args.batchsize, epoch_max=args.epochs,adsbis=args.adsbis)
        # lr_scheduler.step()
        # # Print some statistics inside CSV
        csv_logger.add(acc_train, acc_val, loss_train, loss_val,args.lr)
        csv_logger.write(patience=args.patience,wait=args.wait,choose=args.trans_choose,name=args.name)

        early_stopping(loss_val, model)
        if early_stopping.counter >=args.wait:
            args.lr = adjust_learning_rate(optimizer, args.lr, args.declay)
        if early_stopping.early_stop==True:
            print("Early stopping")
            break


