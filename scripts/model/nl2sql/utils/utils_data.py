import torch
import torch.nn as nn
import nltk
from copy import deepcopy

from matplotlib.pylab import *
from scripts.model.nl2sql.utils.utils import generate_perm_inv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape

    # sort before packking
    l = array(l)
    perm_idx = argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float() # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]

    if return_hidden:
        # hout.shape = [number_of_directoin * num_of_layer, seq_len(=batch size), dim * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode(lstm,
                                   wemb_hpu,
                                   l_hpu,
                                   return_hidden=True,
                                   hc0=None,
                                   last_only=True )

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs


def get_wc1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wc1 = []
    for cond in conds:
        wc1.append(cond[0])
    return wc1


def get_wo1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wo1 = []
    for cond in conds:
        wo1.append(cond[1])
    return wo1


def get_wv1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wv1 = []
    for cond in conds:
        wv1.append(cond[2])
    return wv1


def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sn = []
    g_sc = []
    g_sa = []
    g_wn = []
    g_wc = []
    g_wo = []
    g_wv = []
    g_gn = []
    g_gc = []
    g_hn = []
    g_hc = []
    g_ho = []
    g_hv = []
    g_on = []
    g_oc = []
    g_oa = []
    g_oo = []
    for b, sql in enumerate(sql_i):
        temp = {}
        # SELECT部分
        temp['sel'] = []
        temp['agg'] = []

        table_sel = sql['select']
        gt_sel = table_sel[1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            temp['agg'].append(tup[0])
            temp['sel'].append(tup[1][1][1])

        # WHERE部分
        temp['where'] = []

        table_where = sql['where']
        if len(table_where) > 0:
            # 奇数位是'and'/'or'
            conds = [table_where[x] for x in range(len(table_where)) if x % 2 == 0]
            for cond in conds:
                temp_cond = []

                temp_cond.append(cond[2][1][1])

                temp_cond.append(cond[1])

                if cond[4] is not None:
                    temp_cond.append([cond[3], cond[4]])
                else:
                    temp_cond.append(cond[3])

                temp['where'].append(temp_cond)

        temp['conj'] = [table_where[x] for x in range(len(table_where)) if x % 2 == 1]

        # GROUP BY部分
        temp['group'] = [x[1] for x in sql['groupBy']]  # assume only one groupby
        having_cond = []
        if len(sql['having']) > 0:
            gt_having = sql['having'][0]  # currently only do first having condition
            having_cond.append([gt_having[2][1][0]])  # aggregator
            having_cond.append([gt_having[2][1][1]])  # column
            having_cond.append([gt_having[1]])  # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        temp['group'].append(having_cond)  # GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # ORDER BY部分
        order_aggs = []
        order_cols = []
        temp['order'] = []
        order_par = 4
        gt_order = sql['orderBy']
        limit = sql['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]]  # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        temp['order'] = [order_aggs, order_cols, order_par]  # GOLD for ORDER [[[agg], [col], [dat]], []]

        # process intersect/except/union
        temp['special'] = 0
        if sql['intersect'] is not None:
            temp['special'] = 1
        elif sql['except'] is not None:
            temp['special'] = 2
        elif sql['union'] is not None:
            temp['special'] = 3

        g_sn.append(len(temp['sel']))
        g_sc.append(temp["sel"])
        g_sa.append(temp["agg"])

        where = temp['where']
        g_wn.append(len(where))
        g_wc.append(get_wc1(where))
        g_wo.append(get_wo1(where))
        g_wv.append(get_wv1(where))

    return g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv


def get_g_wvi_nltk(t):
    g_wvi_nltk = []
    for t1 in t:
        conds = t1['where']
        question_toks = t1['question_tok']
        i_temp = []
        for cond in conds:
            op = cond[1]
            if op not in [2, 3, 4, 5, 6, 7]:
                continue
            val = cond[2]
            if type(val).__name__ == "str":
                tokens = nltk.word_tokenize(val)
                st = tokens[1]
                ed = tokens[-2]
                st_idx = question_toks.index(st)
                ed_idx = question_toks.index(ed)
                i_temp.append([st_idx, ed_idx])
            elif type(val).__name__ == 'float':
                if val in question_toks or str(int(val)) in question_toks:
                    if str(int(val)) in question_toks:
                        val = str(int(val))
                    idx = question_toks.index(val)
                    i_temp.append([idx, idx])
        g_wvi_nltk.append(i_temp)
    return g_wvi_nltk


def get_cnt_sn(g_sn, pr_sn):
    cnt = 0
    for b, g_sn1 in enumerate(g_sn):
        pr_sn1 = pr_sn[b]
        if pr_sn1 == g_sn1:
            cnt += 1

    return cnt


def get_cnt_wn(g_wn, pr_wn):
    cnt = 0
    for b, g_wn1 in enumerate(g_wn):
        pr_wn1 = pr_wn[b]
        if pr_wn1 == g_wn1:
            cnt += 1

    return cnt


def get_cnt_sc(g_sc, pr_sc):
    cnt = 0
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt += 1

    return cnt


def get_cnt_sc_list(g_sc, pr_sc):
    cnt_list = []
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_cnt_sa(g_sa, pr_sa):
    cnt = 0
    for b, g_sa1 in enumerate(g_sa):
        pr_sa1 = pr_sa[b]
        if pr_sa1 == g_sa1:
            cnt += 1

    return cnt


def get_cnt_wn(g_wn, pr_wn):
    cnt = 0
    for b, g_wn1 in enumerate(g_wn):
        pr_wn1 = pr_wn[b]
        if pr_wn1 == g_wn1:
            cnt += 1

    return cnt


def get_cnt_wc(g_wc, pr_wc):
    cnt = 0
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()

            if array_equal(pr_wc1, wc1):
                cnt += 1

    return cnt


def get_cnt_wc_list(g_wc, pr_wc):
    cnt_list= []
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            cnt_list.append(0)
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()

            if array_equal(pr_wc1, wc1):
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list


def get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt = 0
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            continue
        else:
            # Sort based on wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))

                g_wo1_s = array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to teacher forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt += 1
    return cnt

def get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt_list=[]
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))

                g_wo1_s = array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to tearch forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list


def get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv

    g_wvi
    """
    cnt = 0
    for b, g_wvi1 in enumerate(g_wvi):
        pr_wvi1 = pr_wvi[b]
        g_wc1 = g_wc[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt += 1

    return cnt


def get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wvi1 in enumerate(g_wvi):
        g_wc1 = g_wc[b]
        pr_wvi1 = pr_wvi[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list


def get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wc1 in enumerate(g_wc):
        pr_wn1 = len(pr_sql_i[b]["conds"])
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi_str11 = str(g_sql_i[b]["conds"][idx11][2]).lower()
                pr_wvi_str11 = str(pr_sql_i[b]["conds"][i_wn][2]).lower()
                # print(g_wvi_str11)
                # print(pr_wvi_str11)
                # print(g_wvi_str11==pr_wvi_str11)
                if g_wvi_str11 != pr_wvi_str11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_sw(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc(g_sc, pr_sc)
    cnt_sa = get_cnt_sa(g_sa, pr_sa)
    cnt_wn = get_cnt_wn(g_wn, pr_wn)
    cnt_wc = get_cnt_wc(g_wc, pr_wc)
    cnt_wo = get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    cnt_wv = get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode)

    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv


def get_cnt_sw_list(g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                    pr_sn, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                    g_sql_i, pr_sql_i,
                    mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sn = get_cnt_sc_list(g_sn, pr_sn)
    cnt_sc = get_cnt_sc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_sc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    if pr_wvi:
        cnt_wvi = get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode)
    else:
        cnt_wvi = [0]*len(cnt_sc)
    cnt_wv = get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode) # compare using wv-str which presented in original data.

    return cnt_sn, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wvi, cnt_wv


def get_cnt_lx_list(cnt_sn1, cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
    # all cnt are list here.
    cnt_list = []
    cnt_lx = 0
    for csc, csa, cwn, cwc, cwo, cwv in zip(cnt_sn1, cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
        if csc and csa and cwn and cwc and cwo and cwv:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_cnt_x_list(engine, tb, g_sc, g_sa, g_sql_i, pr_sc, pr_sa, pr_sql_i):
    cnt_x1_list = []
    g_ans = []
    pr_ans = []
    for b in range(len(g_sc)):
        g_ans1 = engine.execute(tb[b]['id'], g_sc[b], g_sa[b], g_sql_i[b]['conds'])
        # print(f'cnt: {cnt}')
        # print(f"pr_sql_i: {pr_sql_i[b]['conds']}")
        try:
            pr_ans1 = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'])

            if bool(pr_ans1):  # not empty due to lack of the data from incorretly generated sql
                if g_ans1 == pr_ans1:
                    cnt_x1 = 1
                else:
                    cnt_x1 = 0
            else:
                cnt_x1 = 0
        except:
            # type error etc... Execution-guided decoding may be used here.
            pr_ans1 = None
            cnt_x1 = 0
        cnt_x1_list.append(cnt_x1)
        g_ans.append(g_ans1)
        pr_ans.append(pr_ans1)

    return cnt_x1_list, g_ans, pr_ans


def get_g_wvi_bert_from_g_wvi_corenlp(wh_to_wp_index, g_wvi_corenlp):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, g_wvi_corenlp1 in enumerate(g_wvi_corenlp):
        wh_to_wp_index1 = wh_to_wp_index[b]
        g_wvi1 = []
        for i_wn, g_wvi_corenlp11 in enumerate(g_wvi_corenlp1):

            st_idx, ed_idx = g_wvi_corenlp11

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]

            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)

        g_wvi.append(g_wvi1)

    return g_wvi


def get_fields_1(t1, tables, no_hs_t=False, no_sql_t=False):
    nlu1 = t1['question']
    nlu_t1 = t1['question_tok']
    tid1 = t1['table_id']
    sql_i1 = t1['sql']
    sql_q1 = t1['query']
    if no_sql_t:
        sql_t1 = None
    else:
        sql_t1 = t1['query_tok']

    tb1 = tables[tid1]
    if not no_hs_t:
        hs_t1 = tb1['header_tok']
    else:
        hs_t1 = []
    hs1 = tb1['cols']

    return nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1


def get_fields(t1s, tables, no_hs_t=False, no_sql_t=False):

    nlu, nlu_t, tid, sql_i, sql_q, sql_t, tb, hs_t, hs = [], [], [], [], [], [], [], [], []
    for t1 in t1s:
        if no_hs_t:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)
        else:
            nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t)

        nlu.append(nlu1)
        nlu_t.append(nlu_t1)
        tid.append(tid1)
        sql_i.append(sql_i1)
        sql_q.append(sql_q1)
        sql_t.append(sql_t1)

        tb.append(tb1)
        hs_t.append(hs_t1)
        hs.append(hs1)

    return nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hs


def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    hds2 = [i[1] for i in hds1]

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds2):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds


def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx

    return idxs


def check_sc_sa_pairs(tb, pr_sc, pr_sa, ):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    """
    bS = len(pr_sc)
    check = [False] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        pr_sa1 = pr_sa[b]
        hd_types1 = tb[b]['types']
        hd_types11 = hd_types1[pr_sc1]
        if hd_types11 == 'text':
            if pr_sa1 == 0 or pr_sa1 == 3: # ''
                check[b] = True
            else:
                check[b] = False

        elif hd_types11 == 'real':
            check[b] = True
        else:
            raise Exception("New TYPE!!")

    return check


def pred_sc(sn, s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for b, sn1 in enumerate(sn):
        s_sc1 = s_sc[b]

        pr_sc1 = argsort(-s_sc1.data.cpu().numpy())[:sn1]
        pr_sc1.sort()
        pr_sc.append(list(pr_sc1))
    return pr_sc


def pred_sc_beam(s_sc, beam_size):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc_beam = []


    for s_sc1 in s_sc:
        val, idxes = s_sc1.topk(k=beam_size)
        pr_sc_beam.append(idxes.tolist())

    return pr_sc_beam


def pred_sa(sn, s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sa_a = s_sa.argmax(dim=2)  # [B, 4]
    pr_sa = []
    for b, sa in enumerate(pr_sa_a):
        sn1 = sn[b]
        pr_sa.append(list(argsort(sa.data.cpu())[:sn1]))
    return pr_sa


def pred_sn(s_sn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sn = []
    for s_wn1 in s_sn:
        pr_sn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_sn


def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_wn

def pred_wc_old(sql_i, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wc = []
    for b, sql_i1 in enumerate(sql_i):
        wn = len(sql_i1['conds'])
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc

def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc

def pred_wc_sorted_by_prob(s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted by prob.
    All colume-indexes are returned here.
    """
    # get g_num
    bS = len(s_wc)
    pr_wc = []

    for b in range(bS):
        s_wc1 = s_wc[b]
        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())
        pr_wc.append(list(pr_wc1))
    return pr_wc


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op]
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))

    return pr_wo


def pred_wvi_se(wn, s_wv):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    """

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    pr_wvi_st_idx = s_wv_st.argmax(dim=2) # [B, 4, mL] -> [B, 4, 1]
    pr_wvi_ed_idx = s_wv_ed.argmax(dim=2)

    pr_wvi = []
    for b, wn1 in enumerate(wn):
        pr_wvi1 = []
        for i_wn in range(wn1):
            pr_wvi_st_idx11 = pr_wvi_st_idx[b][i_wn]
            pr_wvi_ed_idx11 = pr_wvi_ed_idx[b][i_wn]
            pr_wvi1.append([pr_wvi_st_idx11.item(), pr_wvi_ed_idx11.item()])
        pr_wvi.append(pr_wvi1)

    return pr_wvi

def pred_wvi_se_beam(max_wn, s_wv, beam_size):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx


    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv.shape[0]

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    prob_wv_st = F.softmax(s_wv_st, dim=-1).detach().to('cpu').numpy()
    prob_wv_ed = F.softmax(s_wv_ed, dim=-1).detach().to('cpu').numpy()

    k_logit = int(ceil(sqrt(beam_size)))
    n_pairs = k_logit**2
    assert n_pairs >= beam_size
    values_st, idxs_st = s_wv_st.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]
    values_ed, idxs_ed = s_wv_ed.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]

    # idxs = [B, k_logit, 2]
    # Generate all possible combination of st, ed indices & prob
    pr_wvi_beam = [] # [B, max_wn, k_logit**2 [st, ed] paris]
    prob_wvi_beam = zeros([bS, max_wn, n_pairs])
    for b in range(bS):
        pr_wvi_beam1 = []

        idxs_st1 = idxs_st[b]
        idxs_ed1 = idxs_ed[b]
        for i_wn in range(max_wn):
            idxs_st11 = idxs_st1[i_wn]
            idxs_ed11 = idxs_ed1[i_wn]

            pr_wvi_beam11 = []
            pair_idx = -1
            for i_k in range(k_logit):
                for j_k in range(k_logit):
                    pair_idx += 1
                    st = idxs_st11[i_k].item()
                    ed = idxs_ed11[j_k].item()
                    pr_wvi_beam11.append([st, ed])

                    p1 = prob_wv_st[b, i_wn, st]
                    p2 = prob_wv_ed[b, i_wn, ed]
                    prob_wvi_beam[b, i_wn, pair_idx] = p1*p2
            pr_wvi_beam1.append(pr_wvi_beam11)
        pr_wvi_beam.append(pr_wvi_beam1)


    # prob

    return pr_wvi_beam, prob_wvi_beam


def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index, nlu):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = [] # word-piece version
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str_wp1 = []
        pr_wv_str1 = []
        wp_to_wh_index1 = wp_to_wh_index[b]
        nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]

        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11

            # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
            # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
            pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx+1]
            pr_wv_str_wp1.append(pr_wv_str_wp11)

            st_wh_idx = wp_to_wh_index1[st_idx]
            ed_wh_idx = wp_to_wh_index1[ed_idx]
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx+1]

            pr_wv_str1.append(pr_wv_str11)

        pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)

    return pr_wv_str, pr_wv_str_wp


def pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv):
    pr_sn = pred_sn(s_sn)
    pr_sc = pred_sc(pr_sn, s_sc)
    pr_sa = pred_sa(pr_sn, s_sa)
    pr_wn = pred_wn(s_wn)
    pr_wc = pred_wc(pr_wn, s_wc)
    pr_wo = pred_wo(pr_wn, s_wo)
    pr_wvi = pred_wvi_se(pr_wn, s_wv)

    return pr_sn, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi


def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
               # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()


def generate_sql_i(pr_sn, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        sels = []

        for i_sn in range(pr_sn[b]):
            sels1 = []
            sels1.append(pr_sc[b][i_sn])
            sels1.append(pr_sa[b][i_sn])
            sels.append(sels1)

        for i_wn in range(pr_wn[b]):
            conds1 = []
            conds1.append(pr_wc[b][i_wn])
            conds1.append(pr_wo[b][i_wn])
            merged_wv11 = merge_wv_t1_eng(pr_wv_str[b][i_wn], nlu[b])
            conds1.append(merged_wv11)
            conds.append(conds1)

        pr_sql_i1 = {'sels': sels, 'conds': conds}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i


def sort_pr_wc(pr_wc, g_wc):
    """
    Input: list
    pr_wc = [B, n_conds]
    g_wc = [B, n_conds]


    Return: list
    pr_wc_sorted = [B, n_conds]
    """
    pr_wc_sorted = []
    for b, pr_wc1 in enumerate(pr_wc):
        g_wc1 = g_wc[b]
        pr_wc1_sorted = []

        if set(g_wc1) == set(pr_wc1):
            pr_wc1_sorted = deepcopy(g_wc1)
        else:
            # no sorting when g_wc1 and pr_wc1 are different.
            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted