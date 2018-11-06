import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
import jieba.analyse


def cut_words():
    """
    分词
    :return:
    """
    content_s = []
    for line in contents:
        current_segment = jieba.lcut(line)
        if len(current_segment) > 1 and current_segment != "\r\n":
            content_s.append(current_segment)
    df_content = pd.DataFrame({"content_S": content_s})
    return df_content


def clean_data(df_content):
    """
    数据清洗
    去掉停用词
    :return:
    """
    stopwords = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=["stopword"], encoding="utf-8")
    stop_word_list = stopwords.stopword.values.tolist()
    df_contents = df_content.content_S.values.tolist()
    contents_clean = []
    all_words = []
    for line in df_contents:
        line_clean = []
        for word in line:
            if word in stop_word_list:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


def word_count():
    """
    绘制WordCloud
    :return:
    """
    df_all_words = pd.DataFrame({"all_words": all_word_list})
    words_count = df_all_words.groupby(by=["all_words"])["all_words"].agg({"count": np.size})
    words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
    matplotlib.rcParams["figure.figsize"] = (10.0, 5.0)
    word_cloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=80)
    word_frequency = {x[0]: x[1] for x in words_count.head(100).values}
    word_cloud = word_cloud.fit_words(word_frequency)
    plt.imshow(word_cloud)
    plt.show()


def analyse_words():
    """
    tf-idf 提取关键词？？？
    :return:
    """
    index = 2400
    print(df_news['content'][index])
    content_s_str = "".join(contents_clean_list[index])
    print(" ".join(jieba.analyse.extract_tags(content_s_str, topK=5, withWeight=False)))


if __name__ == '__main__':
    df_news = pd.read_table("val.txt", names=["category", "theme", "URL", "content"], encoding="utf-8")
    df_news = df_news.dropna()
    contents = df_news.content.values.tolist()
    contents_clean_list, all_word_list = clean_data(cut_words())
    df_content_list = pd.DataFrame({"contents_clean": contents_clean_list})
    word_count()
    analyse_words()
