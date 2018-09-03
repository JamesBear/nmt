
import jieba


def get_file_content(file_path, encoding='utf-8'):
    f = open(file_path, 'r', encoding=encoding)
    content = f.read()
    f.close()
    return content

class WordSegmentation:
    def __init__(self, vocab_file_path):
        content = get_file_content(vocab_file_path)
        words = content.splitlines()
        self.vocab = set(words)

    """
    First segment with jieba. For each segmented word, if it's in vocab,
    add it to result; otherwise add each of its character to result.
    """
    def Segment(self, line):
        segs = jieba.cut(line)
        words = []
        for word in segs:
            if word in self.vocab:
                words.append(word)
            else:
                for c in word:
                    words.append(c)
        return words



    