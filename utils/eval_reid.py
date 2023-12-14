# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import pdb


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    
    
    query_arg = np.argsort(q_pids, axis=0)
    result = g_pids[indices]
    gall_re = result[query_arg]
    gall_re = gall_re.astype(np.str)
    # pdb.set_trace()

    result = gall_re[:,:100]
    
    # with open("re.txt", 'w') as file_obj:
    #     for li in result:
    #         for j in range(len(li)):
    #             if j == len(li) - 1:
    #                 file_obj.write(li[j] + "\n")
    #             else:
    #                 file_obj.write(li[j] + " ")
    with open('re.txt', 'w') as f:
        f.write('rank list file\n')


    #pdb.set_trace()
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        q_sceneid = q_sceneids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # original protocol in RGBNT100 or RGBN300
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

        # for each query sample, its gallery samples from same scene with same or neighbour view are discarded # added by zxp
        # symmetrical_cam = (8 - q_camid) % 8
        # remove = (g_pids[order] == q_pid) & ( # same id
        #              (g_sceneids[order] == q_sceneid) & # same scene
        #              ((g_camids[order] == q_camid) | (g_camids[order] == (q_camid + 1)%8) | (g_camids[order] == (q_camid - 1)%8) | # neighbour cam with q_cam
        #              (g_camids[order] == symmetrical_cam) | (g_camids[order] == (symmetrical_cam + 1)%8) | (g_camids[order] == (symmetrical_cam - 1)%8)) # nerighboour cam with symmetrical cam
        #          )
        # new protocol in MSVR310
        remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
        keep = np.invert(remove)

        with open('re.txt', 'a') as f:
            f.write('{}_s{}_v{}:\n'.format(q_pid, q_sceneid, q_camid))
            v_ids = g_pids[order][keep][:max_rank]
            v_cams = g_camids[order][keep][:max_rank]
            v_scenes = g_sceneids[order][keep][:max_rank]
            for vid, vcam, vscene in zip(v_ids, v_cams, v_scenes):
                f.write('{}_s{}_v{}  '.format(vid, vscene, vcam))
            f.write('\n')

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
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
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
