import spacy_stanza
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AlbertTokenizer, ElectraTokenizer

def get_answer_en(passage, question, model):
    # examples
    nlp = spacy_stanza.load_pipeline("en", processors='tokenize,ner')
    nlp.add_pipe('sentencizer')

    sents = []
    nlp_doc = nlp(passage)
    for sent in nlp_doc.sents:
        sents.append(sent.text)

    def split_sent(sent, offset=0, p_flag=True):
        
        nlp_doc = nlp(sent)

        if p_flag:
            sep_token = '[SEP]'
            words, word_start_idx, char_to_word_offset = [sep_token], [0] , []
        else:
            words, word_start_idx, char_to_word_offset = [], [], []
        
        for token in nlp_doc:
            # token match a-b, then split further
            words.append(token.text)

            if p_flag:
                word_start_idx.append(token.idx + len(sep_token) + 1)
            else:
                word_start_idx.append(token.idx)
        
        word_offset = 0
        if p_flag:
            sent_len = len(sent) + len(sep_token) + 1
        else:
            sent_len = len(sent)
        
        for c in range(sent_len):
            if word_offset >= len(word_start_idx)-1 or c < word_start_idx[word_offset+1]: # 마지막 word 이거나(이거 없으면 word_offset+1 에러), 아직 다음 word가 아니면 word_offset 값을 유지  
                char_to_word_offset.append(word_offset + offset)
            else:
                char_to_word_offset.append(word_offset + offset + 1)
                word_offset += 1
        return words, char_to_word_offset, word_start_idx

    question_tokens, ques_char_to_word_offset, ques_word_to_char_idx = split_sent(question, p_flag=False)

    doc_tokens = []
    sent_start_end_position = []
    ctx_text = ""
    ctx_char_to_word_offset = []
    ctx_word_to_char_idx = []
    sep_token = '[SEP]'
    for sent in sents:
        sent += " "
        sent_start_word_id = len(doc_tokens)
        sent_start_char_id = len(ctx_char_to_word_offset)
        cur_sent_words, cur_sent_char_to_word_offset, cur_sent_words_start_idx = split_sent(sent, offset=len(doc_tokens))
        sent = sep_token + ' ' + sent
        ctx_text += sent
        doc_tokens.extend(cur_sent_words)
        ctx_char_to_word_offset.extend(cur_sent_char_to_word_offset)
        
        for cur_sent_word in cur_sent_words_start_idx:
            ctx_word_to_char_idx.append(sent_start_char_id + cur_sent_word)
        assert len(doc_tokens) == len(ctx_word_to_char_idx)

        sent_start_end_position.append((sent_start_word_id, len(doc_tokens)-1))

    #examples -> features
    tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
    max_seq_length = 512
    max_query_length = 50
    cls_token='[CLS]'
    sep_token='[SEP]'

    def _largest_valid_index(spans, limit):
        for idx in range(len(spans)):
            if spans[idx][1] >= limit:
                return idx
        return len(spans)

    all_query_tokens = [cls_token]
    tok_to_orig_index = [-1]
    ques_tok_to_orig_index = [0]
    ques_orig_to_tok_index = []
    ques_orig_to_tok_back_index = []

    for (i, token) in enumerate(question_tokens):
        ques_orig_to_tok_index.append(len(all_query_tokens))

        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            ques_tok_to_orig_index.append(i)
            all_query_tokens.append(sub_token)

        ques_orig_to_tok_back_index.append(len(all_query_tokens) - 1)

    all_query_tokens = all_query_tokens[:max_query_length]
    tok_to_orig_index = tok_to_orig_index[:max_query_length]

    sentence_spans = []
    all_doc_tokens = []
    orig_to_tok_index = []
    orig_to_tok_back_index = []

    all_doc_tokens += all_query_tokens

    for (i, token) in enumerate(doc_tokens): # example.doc_tokens 는 word 로 쪼개짐
        orig_to_tok_index.append(len(all_doc_tokens)) # passage word: q 포함 token index 인데 word 의 첫 token index
        sub_tokens = tokenizer.tokenize(token) # word 를 token 으로 나눔
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i+len(question_tokens)) # query 포함. 같은 숫자는 word 가 여러개의 token 으로 쪼개짐. q+p token: q 포함된 word index, cls, sep 는 -1, [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, ...]
            all_doc_tokens.append(sub_token)

        orig_to_tok_back_index.append(len(all_doc_tokens) - 1) # passage word: q 포함 token index 인데 word 의 마지막 token index, 앞에 0 ~ 17까지는 [CLS] question [SEP], [18, 21, 22, 24, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, ...]

    for sent_span in sent_start_end_position: # query 없음 word 기준 offset, [(0, 20), (21, 48), (49, 64), (65, 81), (82, 89), (90, 100), (101, 112), (113, 130), (131, 162), (163, 173), (174, 180), (181, 186), (187, 236), (237, 242), ...]
        if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
            continue
        sent_start_position = orig_to_tok_index[sent_span[0]]
        sent_end_position = orig_to_tok_back_index[sent_span[1]]
        sentence_spans.append((sent_start_position, sent_end_position)) # q 포함 token 기준, [(18, 45), (46, 90), (91, 113), (114, 134), (135, 142), (143, 155), (156, 172), (173, 195), (196, 232), (233, 246), (247, 253), (254, 259), (260, 318), (319, 324), ...]

    sent_max_index = _largest_valid_index(sentence_spans, max_seq_length-1) # limit (512 토큰) 를 넘지 않는 최대 sentence 개수, 20

    #sentence 개수가 512 token limit 을 넘는 경우, 잘라줌.
    if sent_max_index < len(sentence_spans):
        sentence_spans = sentence_spans[:sent_max_index]
        max_tok_length = sentence_spans[-1][1]

        all_doc_tokens = all_doc_tokens[:max_tok_length]

    # Padding Document
    all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + [sep_token]

    # Get sent token index
    sent_token_idx = []
    for i, token in enumerate(all_doc_tokens):
        if token == '[SEP]' and i != len(all_doc_tokens)-1:
            sent_token_idx.append(i)

    doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)

    query_input_ids = tokenizer.convert_tokens_to_ids(all_query_tokens)

    doc_input_mask = [1] * len(doc_input_ids)
    doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

    doc_pad_length = max_seq_length - len(doc_input_ids)
    doc_input_ids += [0] * doc_pad_length
    doc_input_mask += [0] * doc_pad_length
    doc_segment_ids += [0] * doc_pad_length

    # Padding Question
    query_input_mask = [1] * len(query_input_ids)
    query_segment_ids = [0] * len(query_input_ids)

    query_pad_length = max_query_length - len(query_input_ids)
    query_input_ids += [0] * query_pad_length
    query_input_mask += [0] * query_pad_length
    query_segment_ids += [0] * query_pad_length

    assert len(doc_input_ids) == max_seq_length
    assert len(doc_input_mask) == max_seq_length
    assert len(doc_segment_ids) == max_seq_length
    assert len(query_input_ids) == max_query_length
    assert len(query_input_mask) == max_query_length
    assert len(query_segment_ids) == max_query_length

    context_idxs = torch.LongTensor(1, max_seq_length)
    context_mask = torch.LongTensor(1, max_seq_length)
    segment_idxs = torch.LongTensor(1, max_seq_length)

    query_mapping = torch.Tensor(1, max_seq_length).cuda()

    context_idxs[0].copy_(torch.Tensor(doc_input_ids))
    context_mask[0].copy_(torch.Tensor(doc_input_mask))
    segment_idxs[0].copy_(torch.Tensor(doc_segment_ids))
    if len(sentence_spans) > 0:
        for i in range(sentence_spans[0][0] - 1):
            query_mapping[0, i] = 1

    context_idxs = context_idxs[:1, :max_seq_length].contiguous().cuda()
    context_mask = context_mask[:1, :max_seq_length].contiguous().cuda()
    segment_idxs = segment_idxs[:1, :max_seq_length].contiguous().cuda()
    query_mapping = query_mapping[:1, :max_seq_length].contiguous()

    # inference
    def convert_to_tokens_inference(question_text, ctx_text, question_tokens, doc_tokens, question_word_to_char_idx, ctx_word_to_char_idx, tok_to_orig_index, y1, y2, q_type_prob):
        answer_dict, answer_type_dict = {}, {}
        answer_type_prob_dict = {}
        q_type = np.argmax(q_type_prob, 1)

        def get_ans_from_pos(y1, y2):

            tok_to_orig_map = tok_to_orig_index

            final_text = " "
            if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
                orig_tok_start = tok_to_orig_map[y1]
                orig_tok_end = tok_to_orig_map[y2]

                ques_tok_len = len(question_tokens)
                if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
                    ques_start_idx = question_word_to_char_idx[orig_tok_start]
                    ques_end_idx = question_word_to_char_idx[orig_tok_end] + len(question_tokens[orig_tok_end])
                    final_text = question_text[ques_start_idx:ques_end_idx]
                else:
                    orig_tok_start -= len(question_tokens)
                    orig_tok_end -= len(question_tokens)
                    ctx_start_idx = ctx_word_to_char_idx[orig_tok_start]
                    ctx_end_idx = ctx_word_to_char_idx[orig_tok_end] + len(doc_tokens[orig_tok_end])
                    final_text = ctx_text[ctx_word_to_char_idx[orig_tok_start]:ctx_word_to_char_idx[orig_tok_end]+len(doc_tokens[orig_tok_end])]

            return final_text

        answer_text = ''
        if q_type[0] == 0:
            answer_text = get_ans_from_pos(y1[0], y2[0])
        elif q_type[0] == 1:
            answer_text = 'yes'
        elif q_type[0] == 2:
            answer_text = 'no'
        else: 
            raise ValueError("question type error")

        answer_dict['Question#1'] = answer_text
        answer_type_prob_dict['Question#1'] = q_type_prob[0].tolist()
        answer_type_dict['Question#1'] = q_type[0].item()

        return answer_dict, answer_type_dict, answer_type_prob_dict

    batch = {'context_idxs': context_idxs, 'context_mask': context_mask, 'segment_idxs': segment_idxs, 'sent_token_idx': [sent_token_idx], 'query_mapping': query_mapping}

    with torch.no_grad():
        start, end, q_type, sent, yp1, yp2, _, answer_confidence_score = model(batch, return_yp=True)

    predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

    sup_sents = str()
    sup_sents_no_index = str()
    for i in range(predict_support_np.shape[1]):
        if predict_support_np[0, i] > 0.85:
            sup_sents += " " + str(i) + ": " + sents[i]
            sup_sents_no_index += sents[i] + " "
    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
    answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens_inference(question, ctx_text,
                                                                                question_tokens, doc_tokens,
                                                                                ques_word_to_char_idx,
                                                                                ctx_word_to_char_idx,
                                                                                tok_to_orig_index, 
                                                                                yp1.data.cpu().numpy().tolist(),
                                                                                yp2.data.cpu().numpy().tolist(),
                                                                                type_prob)

    answer_type_dict.update(answer_type_dict_)
    answer_type_prob_dict.update(answer_type_prob_dict_)
    answer_dict.update(answer_dict_)

    for key in answer_dict.keys():
        if '[SEP]' in answer_dict[key]:
            answer_dict[key] = answer_dict[key].replace('[SEP]','')

    return answer_dict['Question#1'], sup_sents, sup_sents_no_index, answer_confidence_score



def get_answer_ko(passage, question, model):

    passage = passage.replace("《", " 《 ").replace("》"," 》 ").replace("(", " ( ").replace(")", " ) ").replace("  "," ").replace("  "," ")
    question = question.replace("《", " 《 ").replace("》"," 》 ").replace("(", " ( ").replace(")", " ) ").replace("  "," ").replace("  "," ")

    # examples
    nlp = spacy_stanza.load_pipeline("ko", processors='tokenize')
    nlp.add_pipe('sentencizer')

    sents = []
    nlp_doc = nlp(passage)
    for sent in nlp_doc.sents:
        sents.append(sent.text)

    def split_sent(sent, offset=0, p_flag=True):
        
        nlp_doc = nlp(sent)

        if p_flag:
            sep_token = '[SEP]'
            words, word_start_idx, char_to_word_offset = [sep_token], [0] , []
        else:
            words, word_start_idx, char_to_word_offset = [], [], []
        
        for token in nlp_doc:
            # token match a-b, then split further
            words.append(token.text)

            if p_flag:
                word_start_idx.append(token.idx + len(sep_token) + 1)
            else:
                word_start_idx.append(token.idx)
        
        word_offset = 0
        if p_flag:
            sent_len = len(sent) + len(sep_token) + 1
        else:
            sent_len = len(sent)
        
        for c in range(sent_len):
            if word_offset >= len(word_start_idx)-1 or c < word_start_idx[word_offset+1]: # 마지막 word 이거나(이거 없으면 word_offset+1 에러), 아직 다음 word가 아니면 word_offset 값을 유지  
                char_to_word_offset.append(word_offset + offset)
            else:
                char_to_word_offset.append(word_offset + offset + 1)
                word_offset += 1
        return words, char_to_word_offset, word_start_idx

    question_tokens, ques_char_to_word_offset, ques_word_to_char_idx = split_sent(question, p_flag=False)

    doc_tokens = []
    sent_start_end_position = []
    ctx_text = ""
    ctx_char_to_word_offset = []
    ctx_word_to_char_idx = []
    sep_token = '[SEP]'
    for sent in sents:
        sent += " "
        sent_start_word_id = len(doc_tokens)
        sent_start_char_id = len(ctx_char_to_word_offset)
        cur_sent_words, cur_sent_char_to_word_offset, cur_sent_words_start_idx = split_sent(sent, offset=len(doc_tokens))
        sent = sep_token + ' ' + sent
        ctx_text += sent
        doc_tokens.extend(cur_sent_words)
        ctx_char_to_word_offset.extend(cur_sent_char_to_word_offset)
        
        for cur_sent_word in cur_sent_words_start_idx:
            ctx_word_to_char_idx.append(sent_start_char_id + cur_sent_word)
        assert len(doc_tokens) == len(ctx_word_to_char_idx)

        sent_start_end_position.append((sent_start_word_id, len(doc_tokens)-1))

    #examples -> features
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    max_seq_length = 512
    max_query_length = 50
    cls_token='[CLS]'
    sep_token='[SEP]'

    def _largest_valid_index(spans, limit):
        for idx in range(len(spans)):
            if spans[idx][1] >= limit:
                return idx
        return len(spans)

    all_query_tokens = [cls_token]
    tok_to_orig_index = [-1]
    ques_tok_to_orig_index = [0]
    ques_orig_to_tok_index = []
    ques_orig_to_tok_back_index = []

    for (i, token) in enumerate(question_tokens):
        ques_orig_to_tok_index.append(len(all_query_tokens))

        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            ques_tok_to_orig_index.append(i)
            all_query_tokens.append(sub_token)

        ques_orig_to_tok_back_index.append(len(all_query_tokens) - 1)

    all_query_tokens = all_query_tokens[:max_query_length]
    tok_to_orig_index = tok_to_orig_index[:max_query_length]

    sentence_spans = []
    all_doc_tokens = []
    orig_to_tok_index = []
    orig_to_tok_back_index = []

    all_doc_tokens += all_query_tokens

    for (i, token) in enumerate(doc_tokens): # example.doc_tokens 는 word 로 쪼개짐
        orig_to_tok_index.append(len(all_doc_tokens)) # passage word: q 포함 token index 인데 word 의 첫 token index
        sub_tokens = tokenizer.tokenize(token) # word 를 token 으로 나눔
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i+len(question_tokens)) # query 포함. 같은 숫자는 word 가 여러개의 token 으로 쪼개짐. q+p token: q 포함된 word index, cls, sep 는 -1, [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, ...]
            all_doc_tokens.append(sub_token)

        orig_to_tok_back_index.append(len(all_doc_tokens) - 1) # passage word: q 포함 token index 인데 word 의 마지막 token index, 앞에 0 ~ 17까지는 [CLS] question [SEP], [18, 21, 22, 24, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, ...]

    for sent_span in sent_start_end_position: # query 없음 word 기준 offset, [(0, 20), (21, 48), (49, 64), (65, 81), (82, 89), (90, 100), (101, 112), (113, 130), (131, 162), (163, 173), (174, 180), (181, 186), (187, 236), (237, 242), ...]
        if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
            continue
        sent_start_position = orig_to_tok_index[sent_span[0]]
        sent_end_position = orig_to_tok_back_index[sent_span[1]]
        sentence_spans.append((sent_start_position, sent_end_position)) # q 포함 token 기준, [(18, 45), (46, 90), (91, 113), (114, 134), (135, 142), (143, 155), (156, 172), (173, 195), (196, 232), (233, 246), (247, 253), (254, 259), (260, 318), (319, 324), ...]

    sent_max_index = _largest_valid_index(sentence_spans, max_seq_length-1) # limit (512 토큰) 를 넘지 않는 최대 sentence 개수, 20

    #sentence 개수가 512 token limit 을 넘는 경우, 잘라줌.
    if sent_max_index < len(sentence_spans):
        sentence_spans = sentence_spans[:sent_max_index]
        max_tok_length = sentence_spans[-1][1]

        all_doc_tokens = all_doc_tokens[:max_tok_length]

    # Padding Document
    all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + [sep_token]

    # Get sent token index
    sent_token_idx = []
    for i, token in enumerate(all_doc_tokens):
        if token == '[SEP]' and i != len(all_doc_tokens)-1:
            sent_token_idx.append(i)

    doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)

    query_input_ids = tokenizer.convert_tokens_to_ids(all_query_tokens)

    doc_input_mask = [1] * len(doc_input_ids)
    doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

    doc_pad_length = max_seq_length - len(doc_input_ids)
    doc_input_ids += [0] * doc_pad_length
    doc_input_mask += [0] * doc_pad_length
    doc_segment_ids += [0] * doc_pad_length

    # Padding Question
    query_input_mask = [1] * len(query_input_ids)
    query_segment_ids = [0] * len(query_input_ids)

    query_pad_length = max_query_length - len(query_input_ids)
    query_input_ids += [0] * query_pad_length
    query_input_mask += [0] * query_pad_length
    query_segment_ids += [0] * query_pad_length

    assert len(doc_input_ids) == max_seq_length
    assert len(doc_input_mask) == max_seq_length
    assert len(doc_segment_ids) == max_seq_length
    assert len(query_input_ids) == max_query_length
    assert len(query_input_mask) == max_query_length
    assert len(query_segment_ids) == max_query_length

    context_idxs = torch.LongTensor(1, max_seq_length)
    context_mask = torch.LongTensor(1, max_seq_length)
    segment_idxs = torch.LongTensor(1, max_seq_length)

    query_mapping = torch.Tensor(1, max_seq_length).cuda()

    context_idxs[0].copy_(torch.Tensor(doc_input_ids))
    context_mask[0].copy_(torch.Tensor(doc_input_mask))
    segment_idxs[0].copy_(torch.Tensor(doc_segment_ids))
    if len(sentence_spans) > 0:
        for i in range(sentence_spans[0][0] - 1):
            query_mapping[0, i] = 1

    context_idxs = context_idxs[:1, :max_seq_length].contiguous().cuda()
    context_mask = context_mask[:1, :max_seq_length].contiguous().cuda()
    segment_idxs = segment_idxs[:1, :max_seq_length].contiguous().cuda()
    query_mapping = query_mapping[:1, :max_seq_length].contiguous()

    # inference
    def convert_to_tokens_inference(doc_tokens, y1, y2, q_type_prob):
        answer_dict, answer_type_dict = {}, {}
        answer_type_prob_dict = {}
        q_type = np.argmax(q_type_prob, 1)

        def get_ans_from_pos(y1, y2):
            final_text = ""
            for i in range(y1, y2 + 1):
                final_text += " " + doc_tokens[i]
            final_text = final_text.replace(" ##", "").replace(" , ", ",")
            final_text = final_text.strip()
            if final_text == '[CLS]':
                final_text = '?'
            return final_text

        answer_text = ''
        if q_type[0] == 0:
            answer_text = get_ans_from_pos(y1[0], y2[0])
        elif q_type[0] == 1:
            answer_text = 'yes'
        elif q_type[0] == 2:
            answer_text = 'no'
        else: 
            raise ValueError("question type error")

        answer_dict['Question#1'] = answer_text
        answer_type_prob_dict['Question#1'] = q_type_prob[0].tolist()
        answer_type_dict['Question#1'] = q_type[0].item()

        return answer_dict, answer_type_dict, answer_type_prob_dict

    batch = {'context_idxs': context_idxs, 'context_mask': context_mask, 'segment_idxs': segment_idxs, 'sent_token_idx': [sent_token_idx], 'query_mapping': query_mapping}

    with torch.no_grad():
        start, end, q_type, sent, yp1, yp2, _, answer_confidence_score = model(batch, return_yp=True)

    predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

    sup_sents = str()
    sup_sents_no_index = str()
    for i in range(predict_support_np.shape[1]):
        if predict_support_np[0, i] > 0.85:
            sup_sents += " " + str(i) + ": " + sents[i]
            sup_sents_no_index += sents[i] + " "
    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
    answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens_inference(all_doc_tokens, 
                                                                                yp1.data.cpu().numpy().tolist(),
                                                                                yp2.data.cpu().numpy().tolist(),
                                                                                type_prob)

    answer_type_dict.update(answer_type_dict_)
    answer_type_prob_dict.update(answer_type_prob_dict_)
    answer_dict.update(answer_dict_)

    for key in answer_dict.keys():
        if '[SEP]' in answer_dict[key]:
            answer_dict[key] = answer_dict[key].replace('[SEP]','')

    return answer_dict['Question#1'], sup_sents, sup_sents_no_index, answer_confidence_score