import os, json
from copy import deepcopy
from matplotlib.pylab import *


import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.model.nl2sql.models.sel_predict import *
from scripts.model.nl2sql.models.cond_predict import *
from scripts.model.nl2sql.utils.utils_data import *
from scripts.model.nl2sql.utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NL2SQL(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, n_order_ops, old=False):
        super(NL2SQL, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_order_ops = n_order_ops
        # where col prediction
        self.snp = SNP(iS, hS, lS, dr)
        # select column prediction
        self.scp = SCP(iS, hS, lS, dr)
        # select agg prediction
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        # where num prediction
        self.wnp = WNP(iS, hS, lS, dr)

        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old) # start-end-search-discriminative model

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs,
                g_sn=None, g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        # sn
        s_sn = self.snp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_sn = g_sn
        else:
            pr_sn = pred_sn(s_sn)

        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)

        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # sa
        s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sn, pr_sc, show_p_sa=show_p_sa)
        if g_sa:
            # it's not necessary though.
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)


        # wn
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # wo
        s_wo = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)

        return s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb,
                     nlu_t, nlu_wp_t, wp_to_wh_index, nlu,
                     beam_size=4,
                     show_p_sc=False, show_p_sa=False,
                     show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        """
        Execution-guided beam decoding.
        """
        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        for i_beam in range(beam_size):
            pr_sc = list( array(pr_sc_beam)[:,i_beam] )
            s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc] # [B]
            prob_sca[:, i_beam, :] =  (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        # First flatten to 1-d
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.

        idxs = remap_sc_idx(idxs, pr_sc_beam) # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1: # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break


        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # Now, Where-clause beam search.
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)
        prob_wc = F.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.max_wn]*bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc) # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b,:] = prob_wc[b,pr_wc_max1]

        # get most probable max_wn where-clouses
        # wo
        s_wo_max = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, show_p_wo=show_p_wo)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, max_wn, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op  in range(self.n_cond_ops-1):
            pr_wo_temp = [ [i_op]*self.max_wn ]*bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, wo=pr_wo_temp, show_p_wv=show_p_wv)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            # pr_wvi_beam = [B, max_wn, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, max_wn, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops-1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops-1): # do not use final one
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.max_wn:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2] ]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn+1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        # s_wv = [B, max_wn, max_nlu_tokens, 2]
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i


def Loss_sn(s_sn, g_sn):
    loss = F.cross_entropy(s_sn, torch.tensor(g_sn).to(device))
    return loss


def Loss_sc(s_sc, g_sc):
    # Construct index matrix
    bS, max_h_len = s_sc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_sc1 in enumerate(g_sc):
        for g_sc11 in g_sc1:
            im[b, g_sc11] = 1.0
    # Construct prob.
    p = F.sigmoid(s_sc)
    loss = F.binary_cross_entropy(p, im)

    return loss


def Loss_sa(s_sa, g_sn, g_sa):
    loss = 0
    for b, g_wn1 in enumerate(g_sn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_sa[b]
        s_wo1 = s_sa[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss


def Loss_wn(s_wn, g_wn):
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))

    return loss


def Loss_wc(s_wc, g_wc):

    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0
    # Construct prob.
    p = F.sigmoid(s_wc)
    loss = F.binary_cross_entropy(p, im)

    return loss


def Loss_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss


def Loss_wv_se(s_wv, g_wn, g_wo, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0
    # g_wvi = torch.tensor(g_wvi).to(device)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        g_wo1 = g_wo[b]

        flag = True

        if g_wn1 == 0:
            flag = False

        # todo: work for comparision
        for i in g_wo1:
            if i not in [2, 3, 4, 5, 6, 7]:
                flag = False

        if flag is not True:
            continue

        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:, 0]
        g_ed1 = g_wvi1[:, 1]
        # loss from the start position
        loss += F.cross_entropy(s_wv[b, :g_wn1, :, 0], g_st1)

        # print("st_login: ", s_wv[b,:g_wn1,:,0], g_st1, loss)
        # loss from the end position
        loss += F.cross_entropy(s_wv[b, :g_wn1, :, 1], g_ed1)
        # print("ed_login: ", s_wv[b,:g_wn1,:,1], g_ed1, loss)

    return loss


def Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_sn(s_sn, g_sn)
    loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sa(s_sa, g_sn, g_sa)
    loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv, g_wn, g_wo, g_wvi)

    return loss
