from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import TestData
from data_manager import *
from eval_metrics import eval_sysu
from model import embed_net
from utils import *
import time 
import scipy.io as scio
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline') # resnet18
parser.add_argument('--resume', '-r', default='sysu_id_drop_lr1.0e-02_dim512all_FPN_resnet50_best.t', type=str, help='resume from checkpoint') #
parser.add_argument('--model_path', default='E:\\wj-lab\\expStad\\save_model\\', type=str, help='model save path')
parser.add_argument('--FPN', default=True, type=str,
                    help='network FPN+IDE test')
parser.add_argument('--re', type=float, default=0.0)
parser.add_argument('--rerank', default=False,type=str, help="perform re-ranking")
parser.add_argument('--log_path', default='E:\\wj-lab\\expStad\\log\\', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--trial', default=0, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(1)

if __name__ == '__main__':
    dataset = args.dataset
    data_path = 'E:\\wj-lab\\data\\SYSU RGB-IR Re-ID\\SYSU-MM01\\'
    n_class = 395
    test_mode = [1, 2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0
    log_path = args.log_path + 'sysu_log/'
    suffix = dataset + '_Result_' + args.arch
    if args.re > 0:
        suffix = suffix + '_re_{}'.format(args.re)
    if args.IDE:
        suffix = suffix + '_IDE'
    if args.FPN:
        suffix = suffix + '_FPN'
    if args.rerank:
        suffix = suffix + '_rerank'

    test_log_file = open(log_path + suffix + '.txt', "w")
    print('parameters ---  pertrained : {} | mode: {} | IDE: {} | FPN: {} |  rerank: {}'.format(args.arch, args.mode, args.IDE,args.FPN,args.rerank), file=test_log_file)
    print('parameters ---  drop: {:.2} | Ir: {:.6} | dim: {}| re: {:.3}'.format(args.drop, args.lr, args.low_dim,args.re), file=test_log_file)
    print('parameters ---  Train_batch_size: {} | Test_batch_size: {} | trial: {}'.format(args.batch_size, args.test_batch, args.trial),file=test_log_file)
    test_log_file.flush()

    print('==> Building model..')
    net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch,IDE=args.IDE,FPN=args.FPN)
    net.to(device)
    cudnn.benchmark = True
    print('==> Resuming from checkpoint..')
    checkpoint_path = args.model_path
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'],strict=False)
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    print('==> Loading data..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])
    end = time.time()

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    feature_dim = args.low_dim
    if args.arch == 'resnet50':
        pool_dim = 2048
    elif args.arch == 'resnet18':
        pool_dim = 512


    def extract_gall_feat(gall_loader):
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = []
        gall_feat_pool = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(tqdm(gall_loader)):
                batch_num = input.size(0)
                input = Variable(input.cuda())      
                feat_pool, feat_pool2, feat, feat2= net(input, input, test_mode[0])
                feat = feat.data
                feat_pool = feat_pool.data
                gall_feat.append(feat) 
                gall_feat_pool.append(feat_pool)          
        gall_feat = torch.cat(gall_feat, 0)
        gall_feat_pool = torch.cat(gall_feat_pool, 0)
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
        return gall_feat, gall_feat_pool


    def extract_query_feat(query_loader):
        net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        query_feat = []
        query_feat_pool = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(tqdm(query_loader)):
                batch_num = input.size(0)
                input = Variable(input.cuda())    
                feat_pool, feat_pool2, feat, feat2 = net(input, input, test_mode[1])
                feat = feat.data
                feat_pool = feat_pool.data 
                query_feat.append(feat)
                query_feat_pool.append(feat_pool)      
        query_feat = torch.cat(query_feat, 0)
        query_feat_pool = torch.cat(query_feat_pool, 0)
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
        return query_feat, query_feat_pool


    def reranking(query_feat, gall_feat, k1=20, k2=6, lamda_value=0.3):
        x = query_feat.view(query_feat.size(0), -1)
        y = gall_feat.view(gall_feat.size(0), -1)
        feat = torch.cat((x, y))
        query_num, all_num = x.size(0), feat.size(0)
        feat = feat.view(all_num, -1)

        dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        dist = dist + dist.t()
        dist.addmm_(1, -2, feat, feat.t())
        dist = dist.cpu()

        original_dist = dist.numpy()
        all_num = original_dist.shape[0]
        original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float16)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        print('starting re_ranking')
        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

        final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist

    def pairwise_distance(query_feat, gall_feat):
        query_feat = query_feat.view(query_feat.size(0), -1)
        gall_feat = gall_feat.view(gall_feat.size(0), -1)

        m, n = query_feat.size(0), gall_feat.size(0)
        distmat = torch.pow(query_feat, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gall_feat, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, query_feat, gall_feat.t())
        distmat = distmat.cpu().numpy()
        return distmat

    query_feat, query_feat_pool = extract_query_feat(query_loader)  

    all_cmc = 0
    all_mAP = 0
    all_cmc_pool = 0

    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        gall_feat, gall_feat_pool = extract_gall_feat(trial_gall_loader) 
   
        if args.rerank:
           distmat = reranking(query_feat, gall_feat)
           distmat_pool = reranking(query_feat_pool, gall_feat_pool)
           cmc, mAP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
           cmc_pool, mAP_pool = eval_sysu(distmat_pool, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            distmat_pool = np.matmul(query_feat_pool,np.transpose(gall_feat_pool))
            cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        print('Test Trial: {}'.format(trial))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        print('mAP: {:.2%}'.format(mAP_pool))


        print('Test Trial: {}'.format(trial),file=test_log_file)
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]),file=test_log_file)
        print('mAP: {:.2%}'.format(mAP),file=test_log_file)
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]),file=test_log_file)
        print('mAP: {:.2%}'.format(mAP_pool),file=test_log_file)
        test_log_file.flush()
    cmc = all_cmc / 10
    mAP = all_mAP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    print('All Average:',file=test_log_file)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]),file=test_log_file)
    print('mAP: {:.2%}'.format(mAP),file=test_log_file)
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]),file=test_log_file)
    print('mAP: {:.2%}'.format(mAP_pool),file=test_log_file)

