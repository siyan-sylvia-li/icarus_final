import math
import time

import torch.nn
from transformers import GPT2Tokenizer, GPT2Model
from icarus_token_aligner import alignment

"""
    Gaussian Mixture Model Implementation
"""


class ModeGPTLSTMHeatmap(torch.nn.Module):
    def __init__(self, n_feat=512, n_layers=2, n_hidden=256, drop_prob=0.1, gpt_embed=768, T=40, mode=0):
        super().__init__()
        # self.conv = torch.nn.Conv1d(n_feat, n_feat, 5, bias=False)
        self.mode = mode
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.lstm = torch.nn.LSTM(n_feat, n_hidden, n_layers, dropout=drop_prob,
                                  batch_first=True)
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.gpt2_model = GPT2Model.from_pretrained('gpt2')
        self.softmax = torch.nn.Softmax(dim=-1)
        self.T = T
        if self.mode == 2:
            n_hidden = 0
        elif self.mode == 1:
            gpt_embed = 0
        # self.mu_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        self.h_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        self.da_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=2)
        # self.sigma_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        # torch.nn.init.xavier_uniform(self.mu_head.weight)
        torch.nn.init.xavier_uniform(self.h_head.weight)
        # torch.nn.init.xavier_uniform(self.sigma_head.weight)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
        return hidden

    def forward(self, x, hidden, trans, w_ind):
        l_out = None
        if self.mode in [0, 1, 3]:
            audio = x.to(self.device)
            """
                Obtain audio embeddings
            """
            # audio = self.conv(audio)
            l_out, l_hidden = self.lstm(audio, hidden)
            # print(l_hidden.shape, "L HIDDEN SHAPE")
        """
            Extract word sequences to be fed into GPT
        """
        if self.mode in [0, 2, 3]:
            tok_seq = self.gpt_tokenizer(trans, padding=True, truncation=True, return_tensors='pt').to(self.device)
            align = alignment(self.gpt_tokenizer, tok_seq, trans).to(self.device)
            gpt_outs = self.gpt2_model(**tok_seq).last_hidden_state
            gpt_select = []
            for i in range(w_ind.shape[0]):
                align_inds = torch.index_select(align[i], dim=0, index=w_ind[i])
                gpt_select.append(torch.index_select(gpt_outs[i, :, :].unsqueeze(0), dim=1, index=align_inds))
            gpt_select = torch.cat(gpt_select, dim=0)

            if self.mode in [0, 3]:
                l_out = torch.cat([l_out, gpt_select], dim=2)
            else:
                l_out = gpt_select

        """
            Concatenate the outputs
        """
        h = self.h_head(l_out)
        h = self.softmax(h)
        # da = self.da_head(l_out)
        # da = self.softmax(da)

        return h

    def forward_real(self, x, hidden, trans):
        l_out = None
        timing_dict = {}
        timing_dict.update({"LSTM started": time.time()})
        if self.mode in [0, 1]:
            audio = x.to(self.device)
            """
                Obtain audio embeddings
            """
            # audio = self.conv(audio)
            l_out, hidden = self.lstm(audio, hidden)
            # print(l_hidden.shape, "L HIDDEN SHAPE")
        """
            Extract word sequences to be fed into GPT
        """
        timing_dict.update({"LSTM completed": time.time()})
        timing_dict.update({"GPT-2 started": time.time()})
        if self.mode in [0, 2, 3]:
            tok_seq = self.gpt_tokenizer(trans, padding=True, truncation=True, return_tensors='pt').to(self.device)
            # print(self.gpt_tokenizer.batch_decode(tok_seq['input_ids']))
            gpt_outs = self.gpt2_model(**tok_seq).last_hidden_state
            gpt_select = gpt_outs[0, -1, :].unsqueeze(0).unsqueeze(0)

            if self.mode in [0, 3]:
                l_out = torch.cat([l_out, gpt_select], dim=2)
            else:
                l_out = gpt_select
        timing_dict.update({"GPT-2 completed": time.time()})
        timing_dict.update({"Inference started": time.time()})
        """
            Concatenate the outputs
        """
        h = self.h_head(l_out)
        h = self.softmax(h)
        h_ind = torch.argmax(h, dim=-1)
        h_pred = (h_ind / h.shape[-1]) * 2 * 2

        if hasattr(self, 'da_head'):
            da = self.da_head(l_out)
            da = self.softmax(da)
            da_ind = torch.argmax(da, dim=-1)
        else:
            da_ind = 0

        timing_dict.update({"Inference completed": time.time()})

        return h_pred, da_ind, hidden, timing_dict
