import re

import pandas
import glob
import pickle
import matplotlib.pyplot as plt

# all_tags_nums = dict()
all_tags_nums = pickle.load(open("tag_counts_collapsed.p", "rb"))
# all_tags_df = pandas.DataFrame([], columns=['act_tag', 'text', 'text_len'])
all_tags_df = pandas.read_csv('tag_counts_collapsed.csv')
# all_tags_utt = dict()
all_tags_utt = pickle.load(open("tag_utt_collapsed.p", "rb"))
all_tag_dict = dict()


def process_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text).lower().split(" ")
    text = [t for t in text if (t == 'a' or t == 'i' or len(t) > 1)]
    return len(text)


def process_text_clean(text):
    text = re.sub(r'[^a-zA-Z0-9 ]+', '', text).lower().split(" ")
    text = [t for t in text if (t == 'a' or t == 'i' or len(t) > 1)]
    return " ".join(text)


def process_tag(tag: str):
    orig_tag = tag
    # Split by comma and only take first tag
    if "," in tag:
        tag = tag.split(",")[0]
    if "^" in tag and tag.index("^") > 0:
        tag = tag[:tag.index("^")]
    # Truncate anything non-alphabetic
    m = [k.start(0) for k in re.finditer(r'[^a-zA-Z]', tag)]
    m = [x for x in m if x > 0]
    if len(m):
        tag = tag[:m[0]]
    if tag not in all_tag_dict:
        all_tag_dict[tag] = []
    all_tag_dict[tag].append(orig_tag)
    return tag


# def plot_graph():
def extract_data():
    global all_tags_df, all_tags_nums
    for f in glob.glob("swda/swda/swda/*/*.csv"):
        print(f)
        df = pandas.read_csv(f)
        rel_df = df[['act_tag', 'text']].copy()
        rel_df['text_len'] = rel_df.apply(lambda x: process_text(x['text']), axis=1)
        rel_df['text_clean'] = rel_df.apply(lambda x: process_text_clean(x['text']), axis=1)
        rel_df['act_tag'] = rel_df.apply(lambda x: process_tag(x['act_tag']), axis=1)
        df['update_act_tag'] = rel_df['act_tag'].copy()
        df.to_csv(f, index=False)
        del df
        # print(rel_df)
        """
            Updating the Dictionary
        """

        for i, row in rel_df.iterrows():
            if row['act_tag'] not in all_tags_nums:
                all_tags_nums[row['act_tag']] = []
                all_tags_utt[row['act_tag']] = []
            all_tags_nums[row['act_tag']].append(row['text_len'])
            all_tags_utt[row['act_tag']].append(row['text_clean'])

        """
            Updating the Pandas DF
        """
        all_tags_df = all_tags_df.append(rel_df)
        del rel_df
    pickle.dump(all_tags_nums, open("tag_counts_collapsed.p", "wb+"))
    pickle.dump(all_tags_utt, open("tag_utt_collapsed.p", "wb+"))
    pickle.dump(all_tag_dict, open("tag_dict.p", "wb+"))
    all_tags_df.to_csv('tag_counts_collapsed.csv', index=False)


if __name__ == "__main__":
    # Read csv file into data frame
    extract_data()
    # input("Extraction Complete")
    # sorter = [(a, len(all_tags_nums[a])) for a in all_tags_nums]
    # sorter.sort(key=lambda x: x[1], reverse=True)
    # sort_tags = [s[0] for s in sorter]
    # sorter = [s[0] + " [" + str(s[1]) + "]" for s in sorter]
    #
    # print(all_tags_df)
    #
    # fig, ax = plt.subplots()
    # fig.set_size_inches(40, 10)
    # plt.suptitle('')
    # all_tags_df['act_tag'] = pandas.Categorical(all_tags_df['act_tag'], categories=sort_tags, ordered=True)
    # boxplot = all_tags_df.boxplot(by='act_tag', return_type='axes', figsize=(40, 10), rot=45,
    #                               fontsize=12, ax=ax)
    # ax.set_xticklabels(sorter)
    # # plt.sca(ax)
    #
    # plt.plot([0, len(sorter)], [3, 3], color='r', linestyle='-', linewidth=1)
    # print(plt.subplots())
    #
    # plt.show()

    tgs = pickle.load(open("tag_counts_collapsed.p", "rb"))
    tg_file = open("dialogue_acts.txt", "w+")
    for t in sorted(list(tgs.keys())):
        tg_file.write(t + "\n")
    print(tgs.keys())
    print(len(tgs))

