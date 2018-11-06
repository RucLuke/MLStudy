import jieba
import gensim
import matplotlib
import numpy as np
import pandas as pd
import jieba.analyse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from gensim import corpora


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


def lda_model():
    """
    LDA 主题模型
    :return:
    """
    dictionary = corpora.Dictionary(contents_clean_list)
    corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean_list]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    for topic in (lda.print_topics(num_topics=20, num_words=5)):
        print(topic[1])


def content_label():
    """
    利用贝叶斯进行文本分类
    :return:
    """
    df_train = pd.DataFrame({"contents_clean": contents_clean_list, "label": df_news['category']})
    # print(df_train.label.unique())
    label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
    df_train['label'] = df_train['label'].map(label_mapping)
    x_train, x_test, y_train, y_test = train_test_split(df_train["contents_clean"].values,
                                                        df_train["label"].values.ravel())
    words = []
    for line_index in range(len(x_train)):
        try:
            words.append(' '.join(x_train[line_index]))
        except:
            print(line_index, words)
    print(words[0])
    vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vec.fit(words)
    classifier = MultinomialNB()
    classifier.fit(vec.transform(words), y_train)
    test_words = []
    for line_index in range(len(x_test)):
        try:
            test_words.append(' '.join(x_test[line_index]))
        except:
            print(line_index)
    score = classifier.score(vec.transform(test_words), y_test)
    print(score)

    vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vectorizer.fit(words)
    classifier = MultinomialNB()
    classifier.fit(vectorizer.transform(words), y_train)
    score = classifier.score(vectorizer.transform(test_words), y_test)
    print(score)


def count_vector():
    """
    CV
    :return:
    """
    texts = ["dog cat fish", "dog cat cat", "fish bird", "bird"]
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(texts)
    print(cv.get_feature_names())
    print(cv_fit.toarray())
    print(cv_fit.toarray().sum(axis=0))


if __name__ == '__main__':
    df_news = pd.read_table("val.txt", names=["category", "theme", "URL", "content"], encoding="utf-8")
    df_news = df_news.dropna()
    contents = df_news.content.values.tolist()
    contents_clean_list, all_word_list = clean_data(cut_words())
    df_content_list = pd.DataFrame({"contents_clean": contents_clean_list})
    # word_count()
    # analyse_words()
    # lda_model()
    content_label()
    # count_vector()
