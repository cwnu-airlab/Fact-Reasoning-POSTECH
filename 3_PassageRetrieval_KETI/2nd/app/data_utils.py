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


import itertools

from torch.utils.data._utils.collate import default_convert, default_collate

# from facebook research mdr github
def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_tokens_new_dim(values, pad_idx):
    """[bsz, cand_num, seq_len] --> [bsz, cand_num, max_seq_len]"""
    size = max(v.size(0) for v in itertools.chain(*values))
    cand_num = min([len(v) for v in values])
    bsz = len(values)
    res = values[0][0].new(bsz, cand_num, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i in range(bsz):
        for ii in range(cand_num):
            copy_tensor(values[i][ii], res[i][ii][:len(values[i][ii])])
    return res


def collate_new_dim(values):
    """[bsz, cand_num, ?] --> [bsz, cand_num, ?]"""
    cand_num = min([len(v) for v in values])
    new_value = [v[:cand_num] for v in values]
    return default_collate(new_value)


