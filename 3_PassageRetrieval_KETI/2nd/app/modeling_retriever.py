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
import os
import copy
from typing import Callable, Dict, Type
import importlib

from transformers import AutoModel, T5EncoderModel, T5Config
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)

import torch.nn as nn
import torch
import torch.nn.functional as F


def simple_pooling(hidden_states, mask=None):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq]
    first_token_tensor = hidden_states[:, :1]
    
    # pooled_output: [batch_size, 1, model_dim]
    return first_token_tensor


def mean_pooling(hidden_states, mask=None, sqrt=True):
    # hidden states: [batch_size, seq, model_dim]
    # attention masks: [batch_size, seq]
    
    if mask is None:
        batch_size, seq_length = hidden_states.shape[:2]
        mask = torch.ones(batch_size, seq_length, device=hidden_states.device, dtype=hidden_states.dtype)

    sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)).permute(0, 2, 1)
    # sentence_sums: [batch_size, 1, model_dim]
    divisor = mask.sum(dim=1).view(-1, 1, 1).float()
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    # pooled_output: [batch_size, 1, model_dim]
    return sentence_sums


class SimplePooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        pooled_output = simple_pooling(hidden_states, mask)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, model_dim]
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask=None, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq]
        pooled_output = mean_pooling(hidden_states, mask, sqrt=sqrt)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, model_dim]
        return pooled_output


class MeanReducer(nn.Module):
    def __init__(self, hidden_size, repr_size, dropout_rate=0.0, layer_norm_eps=1e-12):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, repr_size)
        self.layernorm = nn.LayerNorm(repr_size, eps=layer_norm_eps)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, repr_size]
        # attention masks: [batch_size, seq]
        pooled_output = mean_pooling(hidden_states, mask, sqrt=sqrt)

        pooled_output = self.dense1(pooled_output)
        pooled_output = self.layernorm(pooled_output)
        
        # pooled_output: [batch_size, 1, repr_size]
        return pooled_output


class T5EncoderSimple(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderSimple, self).__init__(config)

        self.pooler = SimplePooler(config.d_model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
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

        hidden_states = outputs[0]
        pooled_output = self.pooler(hidden_states, attention_mask)

        all_hidden_states = outputs[1] if output_hidden_states else None
        all_attentions = outputs[2] if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class T5EncoderMean(T5EncoderModel):
    def __init__(self, config):
        super(T5EncoderMean, self).__init__(config)

        self.pooler = MeanPooler(config.d_model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
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

        hidden_states = outputs[0]
        pooled_output = self.pooler(hidden_states, attention_mask)

        all_hidden_states = outputs[1] if output_hidden_states else None
        all_attentions = outputs[2] if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    pooled_output,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )




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
            self.load_weight_from_args(args)

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
    
    def load_weight_from_args(self, args):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        outputs = self.encoder_q(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs[1]
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        outputs = self.encoder_k(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs[1]

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
        query = self.encode_query(batch['query'])

        with torch.no_grad():
            pos_ctx = self.encode_context(batch['pos_ctx'])
            neg_ctx = self.encode_context(batch['neg_ctx'])
        
        vectors = {'query': query, 'pos_ctx': pos_ctx, "neg_ctx": neg_ctx,}
        return vectors


class T5SimpleMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(T5SimpleMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderSimple.from_pretrained(args.config_path)
        self.encoder_k = T5EncoderSimple.from_pretrained(args.config_path)

    def load_weight_from_args(self, args):
        if os.path.isdir(args.pre_trained_model):
            model_state_dict = T5EncoderSimple.from_pretrained(args.pre_trained_model)
        else:
            model_state_dict = T5EncoderModel.from_pretrained(args.pre_trained_model)
        model_state_dict = model_state_dict.state_dict()

        self.encoder_q.load_state_dict(model_state_dict, strict=False)
        self.encoder_k.load_state_dict(model_state_dict, strict=False)


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderSimple.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderSimple.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


class T5MeanMomentumRetriever(MomentumRetrieverBase):
    _RETRIEVER_TYPE='momentum'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None,
                k=38400,
                m=0.99):
        super(T5MeanMomentumRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k,
            k=k,
            m=m
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderMean.from_pretrained(args.config_path)
        self.encoder_k = T5EncoderMean.from_pretrained(args.config_path)

    def load_weight_from_args(self, args):
        if os.path.isdir(args.pre_trained_model):
            model_state_dict = T5EncoderMean.from_pretrained(args.pre_trained_model)
        else:
            model_state_dict = T5EncoderModel.from_pretrained(args.pre_trained_model)
        model_state_dict = model_state_dict.state_dict()

        self.encoder_q.load_state_dict(model_state_dict, strict=False)
        self.encoder_k.load_state_dict(model_state_dict, strict=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderMean.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderMean.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)



# =========================================================================
# Bi encoder
# =========================================================================

class BiEncoderRetrieverBase(nn.Module):
    _RETRIEVER_TYPE='biencoder'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None):
        super(BiEncoderRetrieverBase, self).__init__()

        if args is not None:
            self.create_encoder_from_args(args)
            self.load_weight_from_args(args)

        elif encoder_q is not None and encoder_k is not None:
            self.encoder_q = encoder_q
            self.encoder_k = encoder_k

        else:
            raise ArgumentError("You must pass the args or (encoder_q, encoder_k) as arguments.")
        
        config = self.encoder_q.config

    
    def create_encoder_from_args(self, args):
        raise NotImplementedError
    
    def load_weight_from_args(self, args):
        raise NotImplementedError
    
    def encode_seq(self, inputs):
        outputs = self.encoder_q(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs[1]
    
    def encode_query(self, inputs):
        return self.encode_seq(inputs)
    
    def encode_context(self, inputs):
        outputs = self.encoder_k(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs[1]

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


    def forward(self, batch):
        query = self.encode_query(batch['query'])
        pos_ctx = self.encode_context(batch['pos_ctx'])
        neg_ctx = self.encode_context(batch['neg_ctx'])
        
        vectors = {'query': query, 'pos_ctx': pos_ctx, "neg_ctx": neg_ctx,}
        return vectors


class T5SimpleBiEncoderRetriever(BiEncoderRetrieverBase):
    _RETRIEVER_TYPE='biencoder'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None):
        super(T5SimpleBiEncoderRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderSimple.from_pretrained(args.config_path)
        self.encoder_k = T5EncoderSimple.from_pretrained(args.config_path)

    def load_weight_from_args(self, args):
        if os.path.isdir(args.pre_trained_model):
            model_state_dict = T5EncoderSimple.from_pretrained(args.pre_trained_model)
        else:
            model_state_dict = T5EncoderModel.from_pretrained(args.pre_trained_model)
        model_state_dict = model_state_dict.state_dict()

        self.encoder_q.load_state_dict(model_state_dict, strict=False)
        self.encoder_k.load_state_dict(model_state_dict, strict=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderSimple.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderSimple.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


class T5MeanBiEncoderRetriever(BiEncoderRetrieverBase):
    _RETRIEVER_TYPE='biencoder'

    def __init__(self,
                args=None,
                encoder_q=None,
                encoder_k=None):
        super(T5MeanBiEncoderRetriever, self).__init__(
            args=args,
            encoder_q=encoder_q,
            encoder_k=encoder_k
        )
    
    def create_encoder_from_args(self, args):
        self.encoder_q = T5EncoderMean.from_pretrained(args.config_path)
        self.encoder_k = T5EncoderMean.from_pretrained(args.config_path)

    def load_weight_from_args(self, args):
        if os.path.isdir(args.pre_trained_model):
            model_state_dict = T5EncoderMean.from_pretrained(args.pre_trained_model)
        else:
            model_state_dict = T5EncoderModel.from_pretrained(args.pre_trained_model)
        model_state_dict = model_state_dict.state_dict()

        self.encoder_q.load_state_dict(model_state_dict, strict=False)
        self.encoder_k.load_state_dict(model_state_dict, strict=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        root_path = args[0]
        
        enc_path_q = os.path.join(root_path, "query")
        args_q = copy.deepcopy(list(args))
        args_q[0] = enc_path_q
        encoder_q = T5EncoderMean.from_pretrained(*tuple(args_q), **kwargs)

        enc_path_k = os.path.join(root_path, "key")
        args_k = copy.deepcopy(list(args))
        args_k[0] = enc_path_k
        encoder_k = T5EncoderMean.from_pretrained(*tuple(args_k), **kwargs)

        return cls(encoder_q=encoder_q, encoder_k=encoder_k)


