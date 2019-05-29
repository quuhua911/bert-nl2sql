# -*- coding:utf8 -*-

import argparse
import os
import torch
import torch.utils.data
import random as python_random
import numpy as np
import json
import bert.tokenization as tokenization
import torch.nn as nn
import torch.nn.functional as F

from bert.modeling import BertConfig, BertModel
from scripts.model.nl2sql.nl2sql import *
from scripts.model.nl2sql.utils.utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)

    #args.toy_model = not torch.cuda.is_available()
    args.toy_model = True
    args.toy_size = 12

    return args


# Load data -----------------------------------------------------------------------------------------------
# 加载数据集
def load_dataset(path_data, toy_model=False, toy_size=10):
    print("Loading from datasets...")
    dir = path_data
    # train.json
    TABLE_DIR = os.path.join(dir, "tables.json")
    TRAIN_DIR = os.path.join(dir, "train_spider.json")
    DEV_DIR = os.path.join(dir, "dev_spider.json")
    TEST_DIR = os.path.join(dir, "dev_spider.json")

    with open(TABLE_DIR) as table_inf:
        print("Loading data from %s" % TABLE_DIR)
        table_origin_data = json.load(table_inf)

    train_sql_data, train_table_data, schemas_all = load_dataset_with_format(TRAIN_DIR, table_origin_data, toy_model, toy_size)
    dev_sql_data, dev_table_data, schemas = load_dataset_with_format(DEV_DIR, table_origin_data, toy_model, toy_size)
    test_sql_data, test_table_data, schemas = load_dataset_with_format(TEST_DIR, table_origin_data, toy_model, toy_size)

    TRAIN_DB = 'data/train.db'
    DEV_DB = 'data/dev.db'
    TEST_DB = 'data/test.db'

    # schemas_all = None

    return train_sql_data, train_table_data, dev_sql_data, dev_table_data, \
           test_sql_data, test_table_data, schemas_all, TRAIN_DB, DEV_DB, TEST_DB


# 读取数据集并进行格式化
def load_dataset_with_format(dir, table_origin_data, toy_model, toy_size):
    with open(dir) as dataset_inf:
        print("Loading data from %s" % dir)
        dataset_origin_data = json.load(dataset_inf)

    sql_data, table_data = format_dataset(dataset_origin_data, table_origin_data)

    schemas = {}
    for table in table_origin_data:
        schemas[table["db_id"]] = table

    if toy_model:
        return sql_data[:toy_size], table_data, schemas
    else:
        return sql_data, table_data, schemas


# 格式化数据集
def format_dataset(dataset_origin_data, table_origin_data):
    tables = {}
    table_cols = {}
    sqls = []

    for i in range(len(table_origin_data)):
        table = table_origin_data[i]
        temp= {}
        temp['cols'] = table['column_names']

        db_name = table['db_id']
        table_cols[db_name] = temp
        tables[db_name] = table

    for i in range(len(dataset_origin_data)):
        query = dataset_origin_data[i]
        temp={}

        if "type_of_chart" in query:
            temp["type_of_chart"] = query["type_of_chart"]
            temp["x_col"] = query["x_col"]
            temp["y_col"] = query["y_col"]
        # 单个data基本信息
        temp['question'] = query["question"]
        temp['query'] = query['query']
        temp['question_tok'] = query['question_toks']
        temp['query_tok'] = query['query_toks']
        # 省去了具体值
        # temp['question_tok_concol'] = query['question_tok_concol']
        # temp['question_type_concol_list'] = query['question_type_concol_list']

        temp['table_id'] = query['db_id']

        table = tables[temp['table_id']]
        temp['col_original'] = table["column_names_original"]
        temp['table_original'] = table["table_names_original"]
        temp['foreign_key'] = table['foreign_keys']

        # 处理sql信息
        sql = query['sql']
        temp['sql'] = sql
        # SELECT部分
        temp['sel'] = []
        temp['agg'] = []

        '''
            "select": [
                false, # is distinct
                [
                    [
                        3, # agg op index
                        [
                            0, # saved for col operation
                            [
                                0, # agg op index
                                0, # index of col
                                false # is distinct
                            ],
                            null # second col for col operation
                        ]
                    ]
                ]
            ],
        '''

        table_sel = sql['select']
        gt_sel = table_sel[1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            temp['agg'].append(tup[0])
            temp['sel'].append(tup[1][1][1])

        # WHERE部分
        temp['where'] = []

        '''
            "where": [
                [
                    false,
                    2,
                    [
                        0,
                        [
                            0,
                            13,
                            false
                        ],
                        null
                    ],
                    "\"Yes\"",
                    null
                ]
            ]
        '''

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

        sqls.append(temp)

    return sqls, table_cols


def get_loader_data(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader


def get_data(path_data, args):
    # train_data, train_table, dev_data, dev_table = load_wikisql(path_wikisql, args.toy_model, args.toy_size)
    train_data, train_table, dev_data, dev_table, \
    test_data, test_table, schemas, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(path_data, args.toy_model, args.toy_size)

    train_loader, dev_loader = get_loader_data(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def get_bert(BERT_PT_PATH, bert_type, do_lower_case, no_pretraining):
    # some bert-related params
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    # bert vocab
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    # initial model params
    init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    # print bert_config in detail
    bert_config.print_status()

    model_bert = BertModel(bert_config)
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['none', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists']
    order_ops = ['asc limit 1', 'desc limit 1', 'asc', 'desc']

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    # the code is provided by google bert
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    n_order_ops = len(order_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = NL2SQL(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops, n_order_ops, old=True)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def save_for_evaluation(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )


def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n
    #


def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    doc_tokens = []
    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token

        '''
                    nlu_t1 represents the initial tokenlized nl
                    nlu_tt1 represents the word-pieced tokenlized nl

                    x = tt_to_t_idx1[i] means the i-th object in nlu_tt1 is the x-th object in nlu_t1 
                    y = t_to_tt_idx[i] means the i-th sth in nlu_t1 is the y-th sth in nlu_tt1

        '''

        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))
        # hds1_all_tok = tokenize_hds1(tokenizer, hds1)



        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.

        '''
        
            tokens ： input
            segment_ids1 : the hds and the EOL is 1
            i_nlu1 : (st,ed) of the nlu
            i_hds1 : (st,ed) of each header
            
        '''
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, nlu_tt1, hds1)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        # input / tokens / segment_id / mask
        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        # the pos of the heads and nlu
        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    # the needed input is :
    # input_ids(processed by tokenlization, which is based on tokens)
    # input_mask : The mask has 1 for real tokens and 0 for padding tokens
    # segment_id : represent the situation of the tokens and the end
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    # l_hpu represents the length of all the heads processed by tokenlization
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                # todo: the difference btw wemb_h & wemb_n
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1], :]

    return wemb_h


def get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1):

    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # tokens: BERT intput tokens
    # i_nlu: start and end indices of question in tokens
    # i_hds: start and end indices of headers


    # get the wemb
    wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)

    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)

    return wemb_n, wemb_h, l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0 # count the # of examples
    cnt_sn = 0 # select num
    cnt_sc = 0 # count the # of correct predictions of select column
    cnt_sa = 0 # of selectd aggregation
    cnt_wn = 0 # of where number
    cnt_wc = 0 # of where column
    cnt_wo = 0 # of where operator
    cnt_wv = 0 # of where-value
    cnt_wvi = 0 # of where-value index (on question tokens)
    cnt_
    cnt_lx = 0  # of logical form acc
    cnt_x = 0   # of execution acc

    # Engine for SQL querying.
    # engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        # t is the train data
        # and train_table is the related table
        # todo: the difference btw sql and query?

        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_nltk(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            # transfer from nltk to bert
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        # score
        s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sn=g_sn, g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi)

        # Calculate loss & step
        loss = Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # Calculate gradient
        if iB % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sn, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sn, pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)


        # Cacluate accuracy
        cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sn, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                                   pr_sn, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        # cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sn += sum(cnt_sn1_list)
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        # cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sn = cnt_sn / cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wv / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sn, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out


def ftest(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None # not used
            pr_wv_str=None
            pr_wv_str_wp=None
            loss = torch.tensor([0])



        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)


        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa,g_wn, g_wc,g_wo, g_wvi,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


if __name__ == '__main__':
    # 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    # 2. Paths
    path_h = './'
    path_data = os.path.join(path_h, 'data', 'spider')
    BERT_PT_PATH = path_data

    path_save_for_evaluation = './'

    # 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_data, args)

    # 4. Build & load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    # 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

    # 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    for epoch in range(args.tepoch):
        # train
        # todo: train details
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         args.max_seq_length,
                                         args.num_target_layers,
                                         args.accumulate_gradients,
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_data,
                                         dset_name='train')

        # check DEV
        # todo: test details
        with torch.no_grad():
            acc_dev, results_dev, cnt_list = ftest(dev_loader,
                                                   dev_table,
                                                   model,
                                                   model_bert,
                                                   bert_config,
                                                   tokenizer,
                                                   args.max_seq_length,
                                                   args.num_target_layers,
                                                   detail=False,
                                                   path_db=path_wikisql,
                                                   st_pos=0,
                                                   dset_name='dev', EG=args.EG)
        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')

        # save results for the official evalutaion
        save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')

        # save best model
        # Based on Dev Set logical accuracy lx
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:
            acc_lx_t_best = acc_lx_t
            epoch_best = epoch
            # save best model
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join('.', 'model_best.pt'))

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join('.', 'model_bert_best.pt'))

        print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
