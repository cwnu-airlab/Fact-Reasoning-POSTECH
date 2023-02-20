# Copyright 2021 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentError
from json import encoder
import os
import copy
from abc import abstractclassmethod
from typing import Callable, Dict, Type
import importlib

from torch import embedding
from transformers import AutoModel, T5EncoderModel, RobertaModel, ElectraModel
import torch.nn as nn
import torch
import torch.nn.functional as F


MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str) -> Callable[[Type], Type]:
    """
    Register an model to be available in command line calls.

    >>> @register_model("my_model")
    ... class My_Model:
    ...     pass
    """

    def _inner(cls_):
        global MODEL_REGISTRY
        MODEL_REGISTRY[name] = cls_
        return cls_

    return _inner


def _camel_case(name: str):
    words = name.split('_')
    class_name = ''
    for w in words:
        class_name += w[0].upper() + w[1:]
    return class_name


def load_model(model_path: str):
    global MODEL_REGISTRY
    if model_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_path]
    
    if ':' in model_path:
        path_list = model_path.split(':')
        module_name = path_list[0]
        class_name = _camel_case(path_list[1])
    elif '/' in model_path:
        path_list = model_path.split(':')
        module_path = path_list[0].split('/')
        module_name = '.'.join(module_path)
        class_name = _camel_case(path_list[1])
    else:
        raise ValueError('unsupported model path: {}. '
        'you have to provide full path to model or '
        'register the model using @register_model decorator'.format(model_path))
    
    my_module = importlib.import_module(module_name)
    model_class = getattr(my_module, class_name)
    return model_class


def simple_pooling(hidden_states, mask=None):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq, 1]
    first_token_tensor = hidden_states[:, 0]
    
    return first_token_tensor


def mean_pooling(hidden_states, mask=None, sqrt=True):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq, 1]
    sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).squeeze(-1)
    divisor = mask.sum(dim=1).view(-1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    
    return sentence_sums


class SimplePooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        # self.dense2 = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        pooled_output = simple_pooling(hidden_states, mask)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        # pooled_output = F.relu(pooled_output)
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dense2(pooled_output)
        
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        # self.dense2 = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        pooled_output = mean_pooling(hidden_states, mask, sqrt=sqrt)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        # pooled_output = F.relu(pooled_output)
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dense2(pooled_output)
        
        return pooled_output


# =========================================================================
# single encoder
# =========================================================================

# ================== T5 Encoder ==================

class T5EncoderRetriever(T5EncoderModel):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(T5EncoderRetriever, self).__init__(config)

        if hasattr(self.config, "dropout_rate"):
            self.dropout_rate = self.config.dropout_rate
        else:
            self.dropout_rate = 0

    @property
    def pooler_for_encoder(self):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        return self.encode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_samples(self, batch):
        return self(batch)

    def forward(self, batch):
        c1 = self.encode_seq(batch['start_para_codes'])
        c2 = self.encode_seq(batch['end_para_codes'])

        neg_1 = self.encode_seq(batch['neg_codes_1'])
        neg_2 = self.encode_seq(batch['neg_codes_2'])

        q = self.encode_seq(batch['q_codes'])
        q_sp1 = self.encode_seq(batch['q_sp_codes'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler_for_encoder(last_hidden_state, attention_mask)
        return pooled_output
        

@register_model('T5EncoderVanilaRetriever')
class T5EncoderVanilaRetriever(T5EncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(T5EncoderVanilaRetriever, self).__init__(config)

        self._pooler = simple_pooling
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


@register_model('T5EncoderSimpleRetriever')
class T5EncoderSimpleRetriever(T5EncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(T5EncoderSimpleRetriever, self).__init__(config)

        self._pooler = SimplePooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler
        

@register_model('T5EncoderMeanRetriever')
class T5EncoderMeanRetriever(T5EncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(T5EncoderMeanRetriever, self).__init__(config)

        self._pooler = MeanPooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler

# ================== RoBERTa Encoder ==================

class RobertaEncoderRetriever(RobertaModel):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(RobertaEncoderRetriever, self).__init__(config, add_pooling_layer=False)

        if hasattr(self.config, "hidden_dropout_prob"):
            self.dropout_rate = self.config.hidden_dropout_prob
        else:
            self.dropout_rate = 0

    @property
    def pooler_for_encoder(self):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        return self.encode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_samples(self, batch):
        return self(batch)

    def forward(self, batch):
        c1 = self.encode_seq(batch['start_para_codes'])
        c2 = self.encode_seq(batch['end_para_codes'])

        neg_1 = self.encode_seq(batch['neg_codes_1'])
        neg_2 = self.encode_seq(batch['neg_codes_2'])

        q = self.encode_seq(batch['q_codes'])
        q_sp1 = self.encode_seq(batch['q_sp_codes'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super(RobertaEncoderRetriever, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler_for_encoder(last_hidden_state, attention_mask)
        return pooled_output
        

@register_model('RobertaEncoderVanilaRetriever')
class RobertaEncoderVanilaRetriever(RobertaEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(RobertaEncoderVanilaRetriever, self).__init__(config)

        self._pooler = simple_pooling
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


@register_model('RobertaEncoderSimpleRetriever')
class RobertaEncoderSimpleRetriever(RobertaEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(RobertaEncoderSimpleRetriever, self).__init__(config)

        self._pooler = SimplePooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler
        

@register_model('RobertaEncoderMeanRetriever')
class RobertaEncoderMeanRetriever(RobertaEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(RobertaEncoderMeanRetriever, self).__init__(config)

        self._pooler = MeanPooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler

# ================== Electra Encoder ==================

class ElectraEncoderRetriever(ElectraModel):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(ElectraEncoderRetriever, self).__init__(config)

        if hasattr(self.config, "hidden_dropout_prob"):
            self.dropout_rate = self.config.hidden_dropout_prob
        else:
            self.dropout_rate = 0

    @property
    def pooler_for_encoder(self):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        return self.encode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_samples(self, batch):
        return self(batch)

    def forward(self, batch):
        c1 = self.encode_seq(batch['start_para_codes'])
        c2 = self.encode_seq(batch['end_para_codes'])

        neg_1 = self.encode_seq(batch['neg_codes_1'])
        neg_2 = self.encode_seq(batch['neg_codes_2'])

        q = self.encode_seq(batch['q_codes'])
        q_sp1 = self.encode_seq(batch['q_sp_codes'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super(ElectraEncoderRetriever, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler_for_encoder(last_hidden_state, attention_mask)
        return pooled_output
        

@register_model('ElectraEncoderVanilaRetriever')
class ElectraEncoderVanilaRetriever(ElectraEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(ElectraEncoderVanilaRetriever, self).__init__(config)

        self._pooler = simple_pooling
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


@register_model('ElectraEncoderSimpleRetriever')
class ElectraEncoderSimpleRetriever(ElectraEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(ElectraEncoderSimpleRetriever, self).__init__(config)

        self._pooler = SimplePooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler
        

@register_model('ElectraEncoderMeanRetriever')
class ElectraEncoderMeanRetriever(ElectraEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(ElectraEncoderMeanRetriever, self).__init__(config)

        self._pooler = MeanPooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


# ================== Auto Encoder ==================

class AutoEncoderRetriever(AutoModel):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(AutoEncoderRetriever, self).__init__(config)

        # For BERT or RoBERTa
        # super(AutoEncoderRetriever, self).__init__(config, add_pooling_layer=False)

        self.dropout_rate = 0

    @property
    def pooler_for_encoder(self):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        return self.encode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_samples(self, batch):
        return self(batch)

    def forward(self, batch):
        c1 = self.encode_seq(batch['start_para_codes'])
        c2 = self.encode_seq(batch['end_para_codes'])

        neg_1 = self.encode_seq(batch['neg_codes_1'])
        neg_2 = self.encode_seq(batch['neg_codes_2'])

        q = self.encode_seq(batch['q_codes'])
        q_sp1 = self.encode_seq(batch['q_sp_codes'])
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

    def encode(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super(AutoEncoderRetriever, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler_for_encoder(last_hidden_state, attention_mask)
        return pooled_output
        

@register_model('AutoEncoderVanilaRetriever')
class AutoEncoderVanilaRetriever(AutoEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(AutoEncoderVanilaRetriever, self).__init__(config)

        self._pooler = simple_pooling
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


@register_model('AutoEncoderSimpleRetriever')
class AutoEncoderSimpleRetriever(AutoEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(AutoEncoderSimpleRetriever, self).__init__(config)

        self._pooler = SimplePooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler
        

@register_model('AutoEncoderMeanRetriever')
class AutoEncoderMeanRetriever(AutoEncoderRetriever):
    _RETRIEVER_TYPE='single'

    def __init__(self, config):
        super(AutoEncoderMeanRetriever, self).__init__(config)

        self._pooler = MeanPooler(config.hidden_size, self.dropout_rate)
    
    @property
    def pooler_for_encoder(self):
        return self._pooler


# =========================================================================
# Momentum encoder
# =========================================================================

class MomentumRetrieverBase(nn.Module):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(MomentumRetrieverBase, self).__init__()

        if args is not None:
            self.create_encoder_from_args(args)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.requires_grad = False  # not update by gradient

            # size of memory
            self.k = args.k
            # momentum keep the params of the key encoder
            self.m = args.m
        elif encoder_q is not None and encoder_k is not None:
            self.encoder_q = encoder_q
            self.encoder_k = encoder_k

            # size of memory
            self.k = k
            # momentum keep the params of the key encoder
            self.m = m
        else:
            raise ArgumentError("You must pass the args or (encoder_q, encoder_k) as arguments.")
        
        config = self.encoder_q.config
        self.register_buffer("queue", torch.randn(self.k, config.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def create_encoder_from_args(self, args):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        return self.encoder_q.encode_seq(inputs)
    
    def encode_samples(self, batch):
        return self(batch)
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        return self.encoder_k.encode_seq(inputs)

    def save_pretrained(self, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        self.encoder_q.save_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        self.encoder_k.save_pretrained(*tuple(args_k), **kwargs)


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError
    

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def dequeue_and_enqueue(self, embeddings):
        """
        memory bank of previous context embeddings, c1 and c2
        """
        # gather keys before updating queue
        batch_size = embeddings.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.k:
            batch_size = self.k - ptr
            # discard embeddings that exceed the memory size.
            embeddings = embeddings[:batch_size]

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = embeddings

        ptr = (ptr + batch_size) % self.k  # move pointer
        self.queue_ptr[0] = ptr
        return


    def forward(self, batch):
        q = self.encoder_q.encode_seq(batch['q_codes'])
        q_sp1 = self.encoder_q.encode_seq(batch['q_sp_codes'])

        with torch.no_grad():
            c1 = self.encoder_k.encode_seq(batch['start_para_codes'])
            c2 = self.encoder_k.encode_seq(batch['end_para_codes'])

            neg_1 = self.encoder_k.encode_seq(batch['neg_codes_1'])
            neg_2 = self.encoder_k.encode_seq(batch['neg_codes_2'])
        
        vectors = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
        return vectors

# ================== T5 Encoder ==================

@register_model('T5EncoderVanilaMomentumRetriever')
class T5EncoderVanilaMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(T5EncoderVanilaMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderVanilaRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = T5EncoderVanilaRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderVanilaRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderVanilaRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('T5EncoderSimpleMomentumRetriever')
class T5EncoderSimpleMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(T5EncoderSimpleMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderSimpleRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = T5EncoderSimpleRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderSimpleRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderSimpleRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('T5EncoderMeanMomentumRetriever')
class T5EncoderMeanMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(T5EncoderMeanMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderMeanRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = T5EncoderMeanRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderMeanRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderMeanRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)

# ================== RoBERTa Encoder ==================

@register_model('RobertaEncoderVanilaMomentumRetriever')
class RobertaEncoderVanilaMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(RobertaEncoderVanilaMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = RobertaEncoderVanilaRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = RobertaEncoderVanilaRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = RobertaEncoderVanilaRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = RobertaEncoderVanilaRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('RobertaEncoderSimpleMomentumRetriever')
class RobertaEncoderSimpleMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(RobertaEncoderSimpleMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = RobertaEncoderSimpleRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = RobertaEncoderSimpleRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = RobertaEncoderSimpleRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = RobertaEncoderSimpleRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('RobertaEncoderMeanMomentumRetriever')
class RobertaEncoderMeanMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(RobertaEncoderMeanMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = RobertaEncoderMeanRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = RobertaEncoderMeanRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = RobertaEncoderMeanRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = RobertaEncoderMeanRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)

# ================== Electra Encoder ==================

@register_model('ElectraEncoderVanilaMomentumRetriever')
class ElectraEncoderVanilaMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(ElectraEncoderVanilaMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = ElectraEncoderVanilaRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = ElectraEncoderVanilaRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = ElectraEncoderVanilaRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = ElectraEncoderVanilaRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('ElectraEncoderSimpleMomentumRetriever')
class ElectraEncoderSimpleMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(ElectraEncoderSimpleMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = ElectraEncoderSimpleRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = ElectraEncoderSimpleRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = ElectraEncoderSimpleRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = ElectraEncoderSimpleRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


@register_model('ElectraEncoderMeanMomentumRetriever')
class ElectraEncoderMeanMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(ElectraEncoderMeanMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = ElectraEncoderMeanRetriever.from_pretrained(args.pre_trained_model)
        self.encoder_k = ElectraEncoderMeanRetriever.from_pretrained(args.pre_trained_model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = ElectraEncoderMeanRetriever.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = ElectraEncoderMeanRetriever.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)



