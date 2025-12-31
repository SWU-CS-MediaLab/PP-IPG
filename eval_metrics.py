from __future__ import print_function, absolute_import
import numpy as np
import torch
import time
import os
from torch.autograd import Variable

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP
    
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]

        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def eval_vcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    # print("it is evaluate ing now ")
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_func(net, dataset, test_loader,feat_dim,training_phase,epoch):
    gall_loader,query_loader = test_loader
    if dataset == "RegDB":
        test_mode = [2,1] # visible to thermal
    elif dataset == "SYSU-MM01":
        test_mode = [1,2] #thermal to visible
    elif dataset == "LLCM":
        test_mode =[1,2] # thermal to visible
    elif dataset == "VCM":
        test_mode = [1,2]#thermal to visible
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_label = gall_loader.dataset.test_label
    ngall = len(gall_label)
    gall_feat = np.zeros((ngall, feat_dim))
    idx = np.zeros(ngall)
    sim = np.zeros((ngall, training_phase))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            out = net(input, input, test_mode[0],training_phase)
            feat = out["test"]
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            idx[ptr:ptr + batch_num] = out["idx"].detach().cpu().numpy()
            sim[ptr:ptr + batch_num, :] = out["sim"].detach().cpu().numpy()
            ptr = ptr + batch_num
        log(dataset, training_phase, epoch,idx,sim,"gallery")
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_label = query_loader.dataset.test_label
    nquery = len(query_label)
    query_feat = np.zeros((nquery, feat_dim))
    idx = np.zeros(nquery)
    sim = np.zeros((nquery, training_phase))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            out = net(input, input, test_mode[1],training_phase)
            feat = out["test"]
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            idx[ptr:ptr + batch_num] = out["idx"].detach().cpu().numpy()
            sim[ptr:ptr + batch_num, :] = out["sim"].detach().cpu().numpy()
            ptr = ptr + batch_num
        log(dataset, training_phase, epoch,idx,sim, "query")
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'RegDB':
        cmc, mAP,_ = eval_regdb(-distmat,query_label,gall_label)
    elif dataset == 'SYSU-MM01':
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "LLCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "VCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc,mAP

def log(dataset,training_phase,epoch,idx,sim,mode):
    suffix = "2024.1.23_加上了neutral_loss"
    log_dir = f'./evaluate_log/{suffix}/'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_path = log_dir + f"{training_phase}.txt"
    with open(log_path, 'a') as f:
        # idx = idx.cpu().numpy()
        idx = idx.tolist()
        f.write(f"epoch {epoch} datasets {dataset} {mode}:\r\n")
        strNums = [str(x_i) for x_i in idx]
        str1 = ",".join(strNums)
        f.write(str1)
        f.write('\r\n')
    f.close()

    log_dir = f"./inner_states/{suffix}/"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    log_path = log_dir + f"{training_phase}.txt"
    with open(log_path, 'a') as f:
        mean_sim = sim.mean(axis=0)
        mean_sim = mean_sim.tolist()
        f.write(f"epoch {epoch} datasets:{dataset} {mode}:\r\n")
        strNums = [str(x_i) for x_i in mean_sim]
        str1 = ",".join(strNums)
        f.write(str1)
        f.write('\r\n')
    f.close()

def eval_func2(net, dataset, test_loader,feat_dim,training_phase,epoch = 1, accelerator = None):
    gall_loader,query_loader = test_loader
    # gall_loader,query_loader = accelerator.prepare(gall_loader, query_loader)
    if dataset == "RegDB":
        test_mode = [2,1]
    elif dataset == "SYSU-MM01":
        test_mode = [1,2]
    elif dataset == "LLCM":
        test_mode =[2,1]
    elif dataset == "VCM":
        test_mode = [1,2]
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_label = gall_loader.dataset.test_label
    ngall = len(gall_label)
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[0],training_phase = training_phase)["test"]
            feat = net(input, input, test_mode[0])["test"]

            # '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     gall_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_label = query_loader.dataset.test_label
    nquery = len(query_label)
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[1],training_phase =training_phase)["test"]
            feat = net(input, input, test_mode[1])["test"]

            # '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     query_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'RegDB':
        cmc, mAP,_ = eval_regdb(-distmat,query_label,gall_label)
    elif dataset == 'SYSU-MM01':
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "LLCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "VCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc,mAP


def eval_func2_accelerate(net, dataset, test_loader,feat_dim,training_phase,epoch = 1, accelerator = None):
    gall_loader,query_loader = test_loader
    gall_loader,query_loader = accelerator.prepare(gall_loader, query_loader)
    if dataset == "RegDB":
        test_mode = [2,1] # visible to thermal
    elif dataset == "SYSU-MM01":
        test_mode = [1,2] #thermal to visible
    elif dataset == "LLCM":
        test_mode =[1,2] # thermal to visible
    elif dataset == "VCM":
        test_mode = [1,2]#thermal to visible
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_label = gall_loader.dataset.test_label
    ngall = len(gall_label)
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[0],training_phase = training_phase)["test"]
            feat = net(input, input, test_mode[0])["test"]

            '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     gall_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_label = query_loader.dataset.test_label
    nquery = len(query_label)
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[1],training_phase =training_phase)["test"]
            feat = net(input, input, test_mode[1])["test"]

            '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     query_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'RegDB':
        cmc, mAP,_ = eval_regdb(-distmat,query_label,gall_label)
    elif dataset == 'SYSU-MM01':
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "LLCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    elif dataset == "VCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc,mAP

def eval_func2_AIM(net, dataset, test_loader,feat_dim,training_phase,epoch = 1, accelerator = None):
    gall_loader,query_loader = test_loader
    # gall_loader,query_loader = accelerator.prepare(gall_loader, query_loader)
    if dataset == "RegDB":
        test_mode = [2,1] # visible to thermal
        k1 = 8
        k2 = 2
    elif dataset == "SYSU-MM01":
        test_mode = [1,2] #thermal to visible
        k1 = 20
        k2 = 6
    elif dataset == "LLCM":
        test_mode =[1,2] # thermal to visible
        k1 = 20
        k2 = 6
    elif dataset == "VCM":
        test_mode = [1,2]#thermal to visible
        k1 = 20
        k2 = 6
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_label = gall_loader.dataset.test_label
    ngall = len(gall_label)
    gall_feat = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[0],training_phase = training_phase)["test"]
            feat = net(input, input, test_mode[0])["test"]

            # '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     gall_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_label = query_loader.dataset.test_label
    nquery = len(query_label)
    query_feat = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            if len(input.size()) == 5:
                print(input.size())
                input = input.squeeze(1)
            input = Variable(input.cuda())
            # feat = net(input, input, test_mode[1],training_phase =training_phase)["test"]
            feat = net(input, input, test_mode[1])["test"]

            # '分布式修改位置'
            # feat_all = accelerator.gather(feat)
            # if accelerator.is_local_main_process:
            #     query_feat[ptr:ptr + feat_all.shape[0], :] = feat_all.cpu().numpy()
            #     ptr += feat_all.shape[0]

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    distmat = AIM(torch.from_numpy(query_feat).float().cuda(), torch.from_numpy(gall_feat).float().cuda(), k1=k1, k2=k2)

    distmat = np.argsort(distmat, axis=1)

    # compute the similarity
    # distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    if dataset == 'RegDB':
        # cmc, mAP,_ = eval_regdb(-distmat,query_label,gall_label)
        # cmc = 0
        mAP = get_mAP_WO_cam(distmat, query_label, gall_label)
        cmc = get_cmc_WO_cam(distmat, query_label, gall_label)
    elif dataset == 'SYSU-MM01':
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        # cmc, mAP,_ = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        mAP = get_mAP(distmat, query_label, query_cam, gall_label, gall_cam)
        cmc = get_cmc(distmat, query_label, query_cam, gall_label, gall_cam)
    elif dataset == "LLCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        cmc, mAP,_ = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        mAP = get_mAP(distmat, query_label, query_cam, gall_label, gall_cam)
        cmc = get_cmc_llcm(distmat, query_label, query_cam, gall_label, gall_cam)
    elif dataset == "VCM":
        query_cam = query_loader.dataset.test_cam
        gall_cam = gall_loader.dataset.test_cam
        # cmc, mAP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        mAP = get_mAP(distmat, query_label, query_cam, gall_label, gall_cam)
        cmc = get_cmc_llcm(distmat, query_label, query_cam, gall_label, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))
    return cmc,mAP

def getNewFeature(x, y, k1, k2, mean: bool = False):
    dismat = x @ y.T
    val, rank = dismat.topk(k1)
    dismat[dismat < val[:, -1].unsqueeze(1)] = 0
    if mean:
        dismat = dismat[rank[:, :k2]].mean(dim=1)
    return dismat


def AIM(qf: torch.tensor, gf: torch.tensor, k1, k2):
    qf = qf.to('cuda')
    gf = gf.to('cuda')

    qf = torch.nn.functional.normalize(qf)
    gf = torch.nn.functional.normalize(gf)

    new_qf = torch.concat([getNewFeature(qf, gf, k1, k2)], dim=1)
    new_gf = torch.concat([getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

    new_qf = torch.nn.functional.normalize(new_qf)
    new_gf = torch.nn.functional.normalize(new_gf)

    # additional use of relationships between query sets
    # new_qf = torch.concat([getNewFeature(qf, qf, k1, k2, mean=True), getNewFeature(qf, gf, k1, k2)], dim=1)
    # new_gf = torch.concat([getNewFeature(gf, qf, k1, k2), getNewFeature(gf, gf, k1, k2, mean=True)], dim=1)

    return (-new_qf @ new_gf.T - qf @ gf.T).to('cpu')

def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]
def get_mAP_WO_cam(sorted_indices, query_ids, gallery_ids):
    """
    为RegDB数据集定制版的mAP计算函数
    移除了摄像头ID相关操作

    参数说明:
        sorted_indices: 排序后的结果索引 [query_num, gallery_num]
        query_ids: 查询集ID标签 [query_num]
        gallery_ids: 底库集ID标签 [gallery_num]

    返回:
        mAP: 平均精度均值
    """
    result = gallery_ids[sorted_indices]  # [query_num, gallery_num]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # 删除摄像头过滤操作
        match_i = result[probe_index] == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            # 计算平均精度(AP)
            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    # 计算mAP
    mAP = avg_precision_sum / valid_probe_sample_count if valid_probe_sample_count > 0 else 0.0
    return mAP


def get_cmc_WO_cam(sorted_indices, query_ids, gallery_ids):
    """
    为RegDB数据集定制的CMC计算函数
    移除了摄像头ID相关操作

    参数说明:
        sorted_indices: 排序后的结果索引 [query_num, gallery_num]
        query_ids: 查询集ID标签 [query_num]
        gallery_ids: 底库集ID标签 [gallery_num]

    返回:
        cmc: 累积匹配特性曲线
    """
    # 获取唯一ID数量
    gallery_unique_count = len(np.unique(gallery_ids))
    match_counter = np.zeros(gallery_unique_count)

    result = gallery_ids[sorted_indices]  # [query_num, gallery_num]
    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # 直接使用完整结果（不再过滤摄像头）
        result_i = result[probe_index, :]

        # 移除重复ID（保持顺序）
        result_i_unique = get_unique(result_i)

        # 匹配当前查询ID
        match_i = (result_i_unique == query_ids[probe_index])

        if np.sum(match_i) != 0:  # 如果有匹配
            valid_probe_sample_count += 1
            match_counter += match_i.astype(int)

    # 计算匹配率
    rank = match_counter / valid_probe_sample_count if valid_probe_sample_count > 0 else 0
    cmc = np.cumsum(rank)
    return cmc

def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # remove duplicated id in "stable" manner
        result_i_unique = get_unique(result_i)

        # match for probe i
        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc


def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        # remove gallery samples from the same camera of the probe
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        # remove the -1 entries from the label result
        result_i = np.array([i for i in result_i if i != -1])

        # match for probe i
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:  # if there is true matching in gallery
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP


def get_cmc_llcm(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids, max_rank=20):
    num_q, num_g = sorted_indices.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    cmc = np.zeros(max_rank).astype(np.float32)
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = query_ids[q_idx]
        q_camid = query_cam_ids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = sorted_indices[q_idx]
        remove = (gallery_ids[order] == q_pid) & (gallery_cam_ids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = gallery_ids[order][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]

        # get unique and sorted gallery ids
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        # binary vector, positions with value 1 are correct matches
        new_match = (np.array(new_cmc) == q_pid).astype(np.int32)

        # if no match, skip this query
        if not np.any(new_match):
            continue

        # compute cumulative match curve
        new_cmc_curve = new_match.cumsum()
        new_cmc_curve[new_cmc_curve > 1] = 1  # ensure cmc is 0-1

        # update cmc for current query up to max_rank
        cmc[:len(new_cmc_curve)] += new_cmc_curve[:max_rank]
        num_valid_q += 1.

    if num_valid_q == 0:
        raise RuntimeError("Error: all query identities do not appear in gallery")

    # compute average cmc
    cmc = cmc / num_valid_q
    return cmc