import math

import torch.nn
from transformers import GPT2Tokenizer, GPT2Model
from icarus_token_aligner import alignment
import time
import torch

"""
    Gaussian Mixture Model Implementation
"""


class ModeGPTLSTMGMM(torch.nn.Module):
    def __init__(self, n_feat=512, n_layers=2, n_hidden=128, drop_prob=0.1, gpt_embed=768, T=40, mode=0, gamma=0.999):
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
        self.T = T
        if self.mode == 2:
            n_hidden = 0
        elif self.mode == 1:
            gpt_embed = 0
        self.all_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=3 * T)
        # self.mu_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        # self.h_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        # self.sigma_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=T)
        self.da_head = torch.nn.Linear(in_features=n_hidden + gpt_embed, out_features=2)
        self.softmax = torch.nn.Softmax(dim=-1)
        # torch.nn.init.xavier_uniform(self.mu_head.weight)
        # torch.nn.init.xavier_uniform(self.h_head.weight)
        # torch.nn.init.xavier_uniform(self.sigma_head.weight)
        torch.nn.init.xavier_uniform(self.all_head.weight)
        self.gamma = gamma

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
        return hidden

    def logistic_function(self, inp):
        return (1 / self.gamma) * torch.log(1 + torch.exp(self.gamma * inp))
        #  Only subtract from mu

    def forward(self, x, hidden, trans, w_ind):
        l_out = None
        if self.mode != 2:
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
        if self.mode != 1:
            tok_seq = self.gpt_tokenizer(trans, padding=True, truncation=True, return_tensors='pt').to(self.device)
            align = alignment(self.gpt_tokenizer, tok_seq, trans).to(self.device)
            gpt_outs = self.gpt2_model(**tok_seq).last_hidden_state
            gpt_select = []
            for i in range(w_ind.shape[0]):
                align_inds = torch.index_select(align[i], dim=0, index=w_ind[i])
                gpt_select.append(torch.index_select(gpt_outs[i, :, :].unsqueeze(0), dim=1, index=align_inds))
            gpt_select = torch.cat(gpt_select, dim=0)

            if self.mode != 2:
                l_out = torch.cat([l_out, gpt_select], dim=2)
            else:
                l_out = gpt_select

        """
            Concatenate the outputs
        """

        try:
            all_vec = self.all_head(l_out)
            mu = all_vec[:, :, :self.T]
            sigma = all_vec[:, :, self.T: 2 * self.T]
            h = all_vec[:, :, 2 * self.T:]
            assert mu.shape[-1] == self.T and sigma.shape[-1] == self.T and h.shape[-1] == self.T
        except AttributeError:
            mu = self.mu_head(l_out)
            sigma = self.sigma_head(l_out)
            h = self.h_head(l_out)

        try:
            mu = self.logistic_function(mu)
            sigma = self.logistic_function(sigma)
            h = self.softmax(h)
            # print("H ==>", h, h.shape)
            # assert torch.sum(h[0, 0, :]) == 1
        except AttributeError:
            mu = self.pred_relu(mu)
            sigma = self.pred_relu(sigma)
            h = self.pred_relu(h)

        try:
            da_act = self.da_head(l_out)
            out_dict = {"mu": mu, "sigma": sigma, "h": h, "d_act": da_act}
        except AttributeError:
            out_dict = {"mu": mu, "sigma": sigma, "h": h}

        return out_dict

    def forward_real(self, x, hidden, trans):
        l_out = None
        pred_T = 2 * 16 * 2
        timing_dict = {}
        timing_dict.update({"LSTM started": time.time()})
        with torch.no_grad():
            if self.mode != 2:
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
            if self.mode != 1:
                tok_seq = self.gpt_tokenizer(trans, padding=True, truncation=True, return_tensors='pt').to(self.device)
                # print(self.gpt_tokenizer.batch_decode(tok_seq['input_ids']))
                gpt_outs = self.gpt2_model(**tok_seq).last_hidden_state
                gpt_select = gpt_outs[0, -1, :].unsqueeze(0).unsqueeze(0)

                if self.mode != 2:
                    l_out = torch.cat([l_out, gpt_select], dim=2)
                else:
                    l_out = gpt_select
            timing_dict.update({"GPT-2 completed": time.time()})
            timing_dict.update({"Inference started": time.time()})
            """
                Concatenate the outputs
            """
            all_vec = self.all_head(l_out)
            mu = all_vec[:, :, :self.T]
            sigma = all_vec[:, :, self.T: 2 * self.T]
            h = all_vec[:, :, 2 * self.T:]
            mu = self.logistic_function(mu).squeeze()
            sigma = self.logistic_function(sigma).squeeze()
            h = self.softmax(h).squeeze()
            assert mu.shape[-1] == self.T and sigma.shape[-1] == self.T and h.shape[-1] == self.T

            h_pred = torch.sum(mu * h)

            # i_s = torch.tensor([(j / pred_T) * 2 * 2 for j in range(pred_T)]).to(self.device)
            #
            # gaussians = torch.zeros(self.T, pred_T).to(self.device)
            # for i in range(self.T):
            #     for j in range(pred_T):
            #         g_i = i_s[j]
            #         m, s = mu[i], sigma[i] + 1e-10
            #         gaussians[i][j] = math.exp((-1 / (2 * s * s)) * (m - g_i) * (m - g_i)) * (1 / (s * math.sqrt(2 * math.pi)))
            # # Normalize Gaussians
            # gaussians = torch.divide(gaussians, torch.sum(gaussians, dim=-1, keepdim=True))
            # gaussians = gaussians * h[:, None]
            # gaussians = torch.sum(gaussians * i_s[None, :], dim=-1)
            # h_pred = torch.sum(gaussians, dim=-1)

            if hasattr(self, 'da_head'):
                da = self.da_head(l_out)
                da = self.softmax(da)
                da_ind = torch.argmax(da, dim=-1)
            else:
                da_ind = 0

            timing_dict.update({"Inference completed": time.time()})
            print(h_pred)

            return h_pred, da_ind, hidden, timing_dict
