import re
import collections


def words(text):
    """
    选择出所有的单词
    :param text:
    :return:
    """
    return re.findall('[a-z]+', text.lower())


def train(features):
    """
    训练单词
    :param features:
    :return:
    """
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


def edits1(word):
    """
    编辑距离为1
    :param word:
    :return:
    """
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +
               [word[0:i] + word[i + 1] + word[i + 2:] for i in range(n - 1)] +
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])


def known_edits2(word):
    """
    编辑距离为2
    :param word:
    :return:
    """
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in n_words)


def known(word):
    """
    原有单词
    :param word:
    :return:
    """
    return set(w for w in word if w in n_words)


def correct(word):
    """
    更正单词
    :param word:
    :return:
    """
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    print("known([word]) = %s" % known([word]))
    print("known(edits1(word)) = %s" % known(edits1(word)))
    print("known_edits2(word) = %s" % known_edits2(word))
    print("word = %s" % [word])
    for w in candidates:
        print("word=%s count=%d" % (w, n_words[w]))
    return max(candidates, key=lambda wd: n_words[wd])


if __name__ == '__main__':
    """
    主方法
    """
    n_words = train(words(open("big.txt").read()))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    correction = correct("learn")
    print(correction)
