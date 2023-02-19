from models.layers import *

from pretrained_model.modeling_albert import AlbertModel
from pretrained_model.modeling_electra import ElectraModel

class NoGraphNetwork(nn.Module):

    def __init__(self, config, lang_type):
        super(NoGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        if lang_type == 'ko':
            self.transformer = ElectraModel.from_pretrained(config.model_name_or_path_ko)
            self.config.hidden_dim = 768 #electra
            self.config.input_dim = 768
        else:
            self.transformer = AlbertModel.from_pretrained(config.model_name_or_path_en)
            self.config.hidden_dim = 4096 #albert
            self.config.input_dim = 4096

        self.predict_layer = PredictionLayer(self.config)

        self.sent_limit = self.config.max_sent_num
        self.sent_mlp = OutputLayer(self.config, num_answer=1)

    def forward(self, batch, return_yp):
        
        #Create Attention Mask
        custom_attention_mask_list = []
        batch_size = len(batch['context_idxs'])
        for i in range(batch_size):
            tokens_len = len(batch['context_idxs'][i])
            custom_attention_mask = np.array([[1]*tokens_len]*tokens_len)
            special_token_index = [0] + batch['sent_token_idx'][i] + [tokens_len - 1]
            for j in range(len(special_token_index) - 1):
                for k in range(tokens_len):
                    if k not in special_token_index and not (k > special_token_index[j] and k < special_token_index[j+1]):
                        custom_attention_mask[special_token_index[j]][k] = 0
                        custom_attention_mask[k][special_token_index[j]] = 0
            custom_attention_mask_list.append(custom_attention_mask)

        # create attention mask
        mask_len = len(batch['context_idxs'][0])
        mask_list = []
        for i in range(batch_size):
            zero_pad = nn.ZeroPad2d((0, mask_len - len(custom_attention_mask_list[i]), 0, mask_len - len(custom_attention_mask_list[i])))
            zero_pad_mask = zero_pad(torch.FloatTensor(custom_attention_mask_list[i]).unsqueeze(0).unsqueeze(0).cuda())
            mask_list.append(zero_pad_mask)
        custom_attention_mask = torch.cat(mask_list, 0)
        batch['custom_attention_mask'] = custom_attention_mask

        inputs = {'input_ids':      batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'custom_attention_mask': batch['custom_attention_mask']}  # XLM don't use segment_ids

        context_encoding = self.transformer(**inputs)[0]
        
        query_mapping = batch['query_mapping']

        batch_size = len(batch['context_idxs'])
        sent_mapping = torch.zeros(batch_size, self.sent_limit, len(context_encoding[0])).cuda()
        sent_mapping = create_sent_mapping(batch['sent_token_idx'], sent_mapping)

        sent_representations = torch.bmm(sent_mapping, context_encoding)

        # sent_representations = self.attention_layer(sent_representations)[0] + sent_representations

        sent_logits = self.sent_mlp(sent_representations) # N x max_sent x 1

        sent_logits_aux = Variable(sent_logits.data.new(sent_logits.size(0), sent_logits.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logits], dim=-1).contiguous()

        predictions = self.predict_layer(batch, context_encoding, sent_logits[-1], packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2, max_prob, answer_confidence_score = predictions
            return start, end, q_type, sent_prediction, yp1, yp2, max_prob, answer_confidence_score.item()
        else:
            start, end, q_type = predictions
            return start, end, q_type, sent_prediction
    

def create_sent_mapping(sent_token_idx, sent_mapping): #sent_mapping: batch * sent * token
    for i in range(len(sent_token_idx)):
        for j in range(len(sent_token_idx[i])):
            sent_mapping[i][j][sent_token_idx[i][j]] = 1

    return sent_mapping