import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class OutputLayer(nn.Module):
    def __init__(self, config, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim*2),
            nn.ReLU(),
            BertLayerNorm(config.hidden_dim*2, eps=1e-12),
            nn.Dropout(config.trans_drop),
            nn.Linear(config.hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config

        self.start_linear = OutputLayer(config, num_answer=1)
        self.end_linear = OutputLayer(config, num_answer=1)
        self.type_linear = OutputLayer(config, num_answer=3)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, sent_logits, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        # print("check: ", context_mask[0])
        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # start_prediction: batch*token, context_input: batch*token*300, context_mask:batch*token
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # end_prediction: batch*token
        type_prediction = self.type_linear(context_input[:, 0, :]) # type_prediction: batch*4

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))

        # outer_tmp = outer.view(len(outer),-1)
        # sorted = torch.sort(outer_tmp,descending=True)[1]
        
        # start = []
        # end = []
        # for i in range(len(sorted)):
        #     target = 2 # 0은 가장 큰 수, 1은 두 번째로 큰 수
        #     start.append(sorted[i][target].item()//len(outer[0]))
        #     end.append(sorted[i][target].item()%len(outer[0]))

        # yp1 = torch.LongTensor(start)
        # yp2 = torch.LongTensor(end)

        # if packing_mask is not None:
        #     outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1] #max(dim=2)[0] 각 행별로 가장 큰 값, max(dim=2)[1] 각 행별로 가장 큰 값의 인덱스
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]

        start_logits = F.softmax(outer.max(dim=2)[0], dim=1).squeeze()
        end_logits = F.softmax(outer.max(dim=1)[0], dim=1).squeeze()
        answer_confidence_score = start_logits[yp1] * end_logits[yp2]

        outer_softmax = F.softmax(outer.view(len(outer),-1), dim=1)
        max_prob = outer_softmax.max(dim=1)[0] # outer.max(dim=2)[0].max(dim=1)[0] = outer.max(dim=1)[0].max(dim=1)[0]
        
        return start_prediction, end_prediction, type_prediction, yp1, yp2, max_prob, answer_confidence_score
