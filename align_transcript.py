import glob

import pandas
import re
import os
import shutil
from Levenshtein import distance as lev

mrker = 0
wrongFile = False
errFiles = []


def clean_num(text):
    text = re.sub(r'[^0-9*.]+', '', text).lower()
    return text


def clean_no_split(text):
    text = re.sub(r'\*\[\[.+]]', '', text)
    text = re.sub(r'[^a-zA-Z0-9{<> ]+', '', text).lower()
    text = text.replace("<<", "<")
    text = text.replace(">>", ">")
    while "<" in text and ">" in text:
        ind_l = text.index("<")
        ind_r = text.index(">")
        text = text[: ind_l] + text[ind_r + 1:]
    return text


def process_text_clean(text):
    """
        Process the text so that the cleaned text does not have any brackets,
        as in we:
        1. Get rid of the [] repair elements and {} annotations and /
        segmentations;
        2. Remove the different types of noises marked with <>.
    """
    text = clean_no_split(text)
    text = text.split(" ")
    text = [t for t in text if len(t) and t[0] != "{"]
    return text


def check_mrker(mrker, mrk_ls):
    return mrker < len(mrk_ls) and len(mrk_ls[mrker])


def align(mrk_ls, row):
    global mrker, wrongFile
    if (wrongFile):
        return row
    clean_txt = process_text_clean(row['text'])
    # print("+++++++++++")
    # print(row['text'])
    txt_mrker = 0
    orig_mrker = mrker  # the original value, must not go back
    if len(clean_txt):
        """
            If clean_text is non-empty, there are actually concrete words
        """

        while mrker < len(mrk_ls) and len(mrk_ls[mrker]) and mrk_ls[mrker][-1] != clean_txt[0]:
            mrker = mrker + 1
        try:
            utt_start = float(mrk_ls[mrker][1])
        except ValueError:
            # print(mrk_ls[mrker])
            while check_mrker(mrker, mrk_ls) and mrk_ls[mrker][1] == "*":
                if txt_mrker >= len(clean_txt):
                    row['start_ts'] = 0
                    row['end_ts'] = 0
                    return row
                if lev(mrk_ls[mrker][-1], clean_txt[txt_mrker]) <= 1:
                    txt_mrker += 1
                # print(mrk_ls[mrker], "FINDING")
                mrker = mrker + 1
            if (mrker > len(mrk_ls) - 1):
                row['start_ts'] = 0
                row['end_ts'] = 0
                return row
            try:
                utt_start = float(mrk_ls[mrker][1])
            except IndexError:
                print("Index error!")
                wrongFile = True
                return row
        except IndexError:
            print("Index error!")
            wrongFile = True
            return row
        while txt_mrker < len(clean_txt):
            # print(clean_txt[txt_mrker], mrk_ls[mrker])
            if check_mrker(mrker, mrk_ls) and len(mrk_ls[mrker][-1]) and re.search('[a-zA-Z]',
                                                                                   mrk_ls[mrker][-1]) is not None:
                if lev(clean_txt[txt_mrker], mrk_ls[mrker][-1]) <= 1:
                    mrker += 1
                    txt_mrker += 1
                elif mrk_ls[mrker][1] == "*":
                    mrker += 1
                else:
                    errs = [('their', 'theyre')]
                    if ((clean_txt[txt_mrker], mrk_ls[mrker][-1])) in errs:
                        mrker += 1
                        txt_mrker += 1
                    else:
                        print("ERROR, No match!")
                        wrongFile = True
                        return row
            else:
                mrker += 1

        mrker = mrker - 1
        try:
            utt_end = float(mrk_ls[mrker][1]) + float(mrk_ls[mrker][2])
        except ValueError:
            # print("ValueError when converting")
            while mrk_ls[mrker][1] == "*":
                mrker = mrker - 1
            if mrker > 0 and mrk_ls[mrker][-1] in clean_txt and float(mrk_ls[mrker][1]) > utt_start:
                utt_end = float(mrk_ls[mrker][1]) + float(mrk_ls[mrker][2])
            else:
                utt_end = 0
                utt_start = 0
        except IndexError:
            utt_end = 0
            utt_start = 0

    else:
        utt_start = 0
        utt_end = 0
    # print("The utterance starts at: ", utt_start, ", and ends at: ", utt_end)
    if utt_start > utt_end:
        input("ERROR INCORRECT")
    row['start_ts'] = utt_start
    row['end_ts'] = utt_end
    mrker = mrker + 1
    return row


def read_mrk_csv(csv_file, mrk_file):
    df = pandas.read_csv(csv_file)
    mrk_lines = open(mrk_file, "r").readlines()
    for i in range(len(mrk_lines)):
        mrk_lines[i] = re.sub(r'[\t ]+', '|', mrk_lines[i]).replace("\n", "")
        mrk_lines[i] = [x for x in mrk_lines[i].split('|') if len(x)]
        if len(mrk_lines[i]):
            mrk_lines[i][-1] = clean_no_split(mrk_lines[i][-1])
            mrk_lines[i][1] = clean_num(mrk_lines[i][1])
            mrk_lines[i][2] = clean_num(mrk_lines[i][2])
    df = df.apply(lambda x: align(mrk_lines, x), axis=1)
    df.to_csv(csv_file.replace("utt.csv", "utt.ts.csv"), index=False)


if __name__ == "__main__":
    for c_f in glob.glob('full_data/sw*/sw*.utt.csv'):
        sw_num = re.sub(r'full_data/sw.+/sw', '', c_f).replace(".utt.csv", "")
        print(sw_num)
        # sw_file = glob.glob('swda/swda/swda/sw*utt/sw_*_' + sw_num + ".utt.csv")[0]
        # os.remove(c_f)
        # shutil.copy(sw_file, c_f)
        read_mrk_csv(c_f, c_f.replace('.utt.csv', '.mrk'))
        if (wrongFile):
            errFiles.append(sw_num)
        mrker = 0
        wrongFile = False
    print(wrongFile)
