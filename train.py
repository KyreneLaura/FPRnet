from __future__ import print_function
import argparse
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, TestData
from data_manager import *
from eval_metrics import eval_sysu
from model import embed_net
from utils import *
from tqdm import tqdm
import gc
import numpy as np
import cv2
from plot import Plot
import math
from log import transforms as T

parser = argparse.ArgumentParser(description='PyTorch CamStyle')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--FPN', default=True, type=str,
                    help='network FPN+IDE train')
parser.add_argument('--re', type=float, default=0.0)
parser.add_argument('--rerank', default=False,type=str, help="perform re-ranking")
parser.add_argument('--resume', '-r', default='.t', type=str, help='resume from checkpoint') 
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path',default='E:\\wj-lab\\expStad\\save_model\\',type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='E:\\wj-lab\\expStad\\log\\', type=str,
                    help='log save path') 
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=0, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(0)
    dataset = args.dataset
    data_path = 'E:\\wj-lab\\data\\SYSU RGB-IR Re-ID\\SYSU-MM01\\'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]
    checkpoint_path = args.model_path

   
    if args.method == 'id':
        suffix = dataset + '_id'
    suffix = suffix + '_drop{}'.format(args.drop)
    suffix = suffix + '_lr{:1.1e}'.format(args.lr)
    suffix = suffix + '_dim{}'.format(args.low_dim)
    suffix = suffix + format(args.mode)
    if not args.optim == 'sgd':
        suffix = suffix + '_' + args.optim
    if args.re >0:
        suffix = suffix + '_re{}'.format(args.re)
    if args.IDE:
        suffix = suffix + '_IDE'
    if args.FPN:
        suffix = suffix + '_FPN'
    if args.rerank:
        suffix = suffix + '_rerank'
    suffix = suffix + '_' + args.arch


    test_log_file = open(log_path + suffix + '.txt', "w")
    sys.stdout = Logger(log_path + suffix + '_os.txt')
    print('parameters ---  pertrained : {} | mode: {} | IDE: {} | FPN: {} | rerank: {}'.format(args.arch,args.mode,args.IDE,args.FPN,args.rerank), file=test_log_file)
    print('parameters ---  drop: {:.2} | Ir: {:.6} | dim: {}| re: {:.3}'.format(args.drop, args.lr, args.low_dim, args.re), file=test_log_file)
    print('parameters ---  Train_batch_size: {} | Test_batch_size: {} | trial: {}'.format(args.batch_size, args.test_batch,args.trial),file=test_log_file)
    test_log_file.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_acc = 0 
    start_epoch = 0
    feature_dim = args.low_dim

    print('==> Loading data..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.re == 0:
        transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    else:
        print("==> random erasing ...")
        transform_train = transforms.Compose([
            transforms.ToPILImage(),           
            transforms.Pad(10),       
            transforms.RandomCrop((args.img_h, args.img_w)),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            T.RandomErasing(EPSILON=args.re),
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()

 
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    n_class = len(np.unique(trainset.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
    print('==> Building model..')

    net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch,IDE=args.IDE, FPN=args.FPN)
    net.to(device)
    cudnn.benchmark = True
    size = sum(param.numel() for param in net.parameters())

    print('==> Resuming from checkpoint..')
    checkpoint_path = args.model_path
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'], strict=False)
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    print('Net parameters:', size)
    print(" ==> model size (GB)  ....", (size * 4) / (math.pow(2, 30)))

    if args.method == 'id':
        criterion = nn.CrossEntropyLoss()
        criterion_modality = nn.BCEWithLogitsLoss()
        criterion.to(device)
        criterion_modality.to(device)
   
    ignored_params = list(map(id, net.feature.parameters())) + list(map(id, net.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
   
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.feature.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < 30:
            lr = args.lr
        elif epoch >= 30 and epoch < 60:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        optimizer.param_groups[0]['lr'] = 0.1 * lr
        optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[2]['lr'] = lr
        return lr

    def trainPlus(epoch):
        current_lr = adjust_learning_rate(optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        net.train()
        end = time.time()
        for batch_idx, (input1, input2, label1, label2) in enumerate(tqdm(trainloader)):
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            labels = torch.cat((label1, label2), 0)
            labels = Variable(labels.cuda())
            label1 = Variable(label1.cuda())
            label2 = Variable(label2.cuda())
            labels2_1 = torch.cat((label1, label2), 0)
            labels2_2 = torch.cat((label2, label1), 0)
            labels2 = torch.cat((labels2_1, labels2_2), 0)
            labels2 = torch.cat((labels2, labels2), 0)
            labels2 = Variable(labels2.cuda())

            data_time.update(time.time() - end)   
            outputs,out2, outRGB, outIR, feat, feat2,featRGB, featIR = net(input1, input2)     

            if args.method == 'id':
              
                loss_main = criterion(outputs, labels)
                loss_plus2 = criterion(out2, labels2)   
                loss_RGB = criterion(outRGB, label1)
                loss_IR = criterion(outIR, label2)

                _, predicted1 = outputs.max(1)
                _, predicted2 = out2.max(1)   
                match1 = predicted1.eq(labels).sum().item()
                match2 = predicted2.eq(labels2).sum().item()
               
                ave = (loss_RGB + loss_IR) / 2.0
                cross1 = math.exp(abs(loss_main - ave))
                cross2 = math.exp(abs(loss_plus2 - ave))

                if match2 <= match1:
                       loss_total = loss_main + 1 * loss_RGB + 1 * loss_IR + cross1
                       correct += match1
                       optimizer.zero_grad()
                       loss_total.backward()
                       optimizer.step()
                       train_loss.update(loss_total.item(), 6.5 * input1.size(0))                  
                if match1 < match2:    
                        loss_total = loss_plus2 + 1 * loss_RGB + 1 * loss_IR + cross2
                        correct += match2
                        optimizer.zero_grad()
                        loss_total.backward()
                        optimizer.step()
                        train_loss.update(loss_total.item(), 12.5 * input1.size(0))  

            total += labels.size(0)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 10 == 0:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                      'lr:{} '
                      'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    data_time=data_time, train_loss=train_loss))

        return train_loss.avg

    def testPlus(epoch, rerank=False):
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        gall_feat = []
        gall_feat2 = []
        gall_feat3 = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(tqdm(gall_loader)):
                input = Variable(input.cuda())     
                feat_pool, feat_pool2, feat, feat2 = net(input, input, test_mode[0])  
                feat = feat.data
                gall_feat.append(feat)
        gall_feat = torch.cat(gall_feat, 0)
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
        net.eval()
        print('Extracting Query Feature...')
        start = time.time()

        query_feat = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(tqdm(query_loader)):
                input = Variable(input.cuda())            
                feat_pool, feat_pool2, feat, feat2= net(input, input, test_mode[1])
                feat = feat.data
                query_feat.append(feat)
        query_feat = torch.cat(query_feat, 0)
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()
        if rerank:
            print("==> reranking ....")
            distmat = reranking(query_feat, gall_feat)
            cmc, mAP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat = np.matmul(query_feat, np.transpose(gall_feat))
            cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
        return cmc, mAP


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
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],V[indImages[j], indNonZero[j]])
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

    # training
    print('==> Start Training...') 
    Ploter = Plot('expStad')
    for epoch in range(start_epoch, 81 - start_epoch):
        gc.collect()
        print('==> Preparing Data Loader...')
        # identity sampler
        sampler = IdentitySampler(trainset.train_color_label, \
                                  trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size)
        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, \
                                      sampler=sampler, num_workers=args.workers, drop_last=True)

        if args.FPN:
            losses = trainPlus(epoch)
        if epoch > 0 and epoch % 2 == 0:
            print('Test Epoch: {}'.format(epoch))
            print('Test Epoch: {}'.format(epoch), file=test_log_file)
            if args.FPN:
                cmc, mAP = testPlus(epoch, args.rerank)
            if not args.FPN:
                cmc, mAP = test(epoch, args.rerank)


            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], mAP), file=test_log_file)
            test_log_file.flush()

            Ploter.plot(
                { 'train_loss_Avg':losses,' Rank-1': cmc[0], 'Rank-5': cmc[4],
                  ' Rank-10': cmc[9],' Rank-20': cmc[19],' mAP': mAP
                 },{'epoch':epoch}
            )

            # save model
            if cmc[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc[0]
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            # save model every 20 epochs
            if epoch > 10 and epoch % args.save_epoch == 0:
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))