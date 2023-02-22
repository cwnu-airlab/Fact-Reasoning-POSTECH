import torch
import json
import numpy as np
import os
import shutil
import logging
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from eval.hotpot_evaluate_v1 import eval as hotpot_eval
from csr_mhqa.data_processing import IGNORE_INDEX

logger = logging.getLogger(__name__)

def compute_loss(args, batch, start, end, sent, q_type):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long()) #sent_pred: 80*2, sent_gold: 80

    loss = loss_span + loss_type + loss_sup #+ loss_ent + loss_para

    return loss, loss_span, loss_type, loss_sup #, loss_ent, loss_para


def eval_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}
    prob_dict = {}
    dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.05)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]
    
    for batch in tqdm(dataloader):

        with torch.no_grad():

            start, end, q_type, sent, yp1, yp2, max_prob = model(batch, return_yp=True)

        for i in range(len(batch['ids'])):
            prob_dict.update({batch['ids'][i]:max_prob[i].item()})

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        for key in answer_dict.keys():
            if '[SEP]' in answer_dict[key]:
                answer_dict[key] = answer_dict[key].replace('[SEP]','')

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]): # batch size
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]

            for j in range(predict_support_np.shape[1]): # number of sentences
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i]) #각 shreshold 내 id 별로 supporting fact 를 가져옴.

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold, total_sp_dict, prob_dict

def convert_to_tokens(examples, features, ids, y1, y2, q_type_prob):
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    q_type = np.argmax(q_type_prob, 1)

    def get_ans_from_pos(qid, y1, y2):

        feature = features[qid]
        example = examples[qid]

        tok_to_orig_map = feature.token_to_orig_map

        final_text = " "
        if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
            orig_tok_start = tok_to_orig_map[y1] #question + passage 포함
            orig_tok_end = tok_to_orig_map[y2]

            ques_tok_len = len(example.question_tokens)
            if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len and orig_tok_start != -1 and orig_tok_end != -1: #정답이 question에 있으면
                ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
                ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(example.question_tokens[orig_tok_end])
                final_text = example.question_text[ques_start_idx:ques_end_idx]
            elif orig_tok_start >= ques_tok_len and orig_tok_end >= ques_tok_len: #정답이 passage에 있으면
                orig_tok_start -= len(example.question_tokens) #question token 수만큼 차감
                orig_tok_end -= len(example.question_tokens)
                final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[orig_tok_end]+len(example.doc_tokens[orig_tok_end])]
            else:
                final_text = "?"

        return final_text

    for i, qid in enumerate(ids):
        answer_text = ''
        if q_type[i] == 0:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        else: 
            raise ValueError("question type error")

        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

    return answer_dict, answer_type_dict, answer_type_prob_dict