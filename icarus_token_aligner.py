import torch

"""
    Aligns individual words in a sequece with its corresponding GPT-2 embedding.
    If a word has multiple tokens as decided by the GPT-2 tokenizer,
    Use the last embedding.
"""

def alignment(tokenizer, toks, trans):
    input_ids = toks['input_ids']
    attn_msk = toks['attention_mask']
    gpt_align = torch.zeros(input_ids.shape)

    for i in range(len(input_ids)):
        curr_ind = 0
        for j in range(len(input_ids[i])):
            if attn_msk[i, j]:
                decode = tokenizer.decode(input_ids[i][j])
                if " " in decode:
                    curr_ind += 1
                gpt_align[i, curr_ind] += 1

    gpt_align = torch.cat(
        [torch.sum(gpt_align[:, 0: i + 1], axis=1).view(-1, 1) - 1 for i in range(gpt_align.shape[-1])], axis=1)
    gpt_align[gpt_align == float('-inf')] = -1
    gpt_align = gpt_align.type(torch.IntTensor)

    return gpt_align

