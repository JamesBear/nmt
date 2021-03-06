
import os
from sklearn.model_selection import train_test_split
import jieba

# xiaohuangji.txt config
# DATASET_PATH='D:\\tmp\\xiaohuangji.txt'
# QUES_INDEX=0
# ANS_INDEX=1
# SEPARATER='|'

# gossip.txt config
DATASET_PATH='D:\\tmp\\gossip.txt'
QUES_INDEX=1
ANS_INDEX=2
SEPARATER='#'

OUT_DIR='D:\\tmp\\nmt_data'
QUES_ID='question'
ANS_ID='answer'
TRAIN_FILE_NAME='train'
TEST_FILE_NAME='test'
DEV_FILE_NAME='dev'
VOCAB_FILE_NAME='vocab'
DEFAULT_VOCAB=['<unk>','<s>','</s>']
TRAIN_TEST_SPLIT_RATIO=0.05
MAX_VOCAB_SIZE = 17000

def get_file_content(file_path, encoding='utf-8'):
    f = open(file_path, 'r', encoding=encoding)
    content = f.read()
    f.close()
    return content

def write_to_file(file_path, content,  encoding='utf-8'):
    f = open(file_path, 'w', encoding=encoding)
    f.write(content)
    f.close()

def save_list(file_path, list):
    out_str = '\n'.join(list)
    write_to_file(file_path, out_str)

def word_seg(line, vocab):
    words = list(jieba.cut(line))
    for w in words:
        #vocab.add(w)
        count = vocab.get(w)
        if count == None:
            vocab[w] = 1
        else:
            vocab[w] = count+1
    #line = ' '.join(words)
    return words

def resegment_sentences(sentence_list, vocab):
    for i in range(len(sentence_list)):
        words = sentence_list[i]
        segs = []
        for w in words:
            if w in vocab:
                segs.append(w)
            else:
                for c in w:
                    segs.append(c)
        sentence_list[i] = ' '.join(segs)
        

def process(file_path):
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    content = get_file_content(file_path)
    questions = []
    answers = []
    vocab = dict()
    # get question list and answer list
    for line in content.splitlines():
        line = line.strip()
        if line == '':
            continue
        splitted = line.split(SEPARATER)
        if len(splitted) <= max(QUES_INDEX, ANS_INDEX):
            continue

        question = splitted[QUES_INDEX]
        answer = splitted[ANS_INDEX]
        
        if question.strip() == '' or answer.strip() == '':
            #print('empty:', question, answer)
            continue
        question = word_seg(question, vocab)
        answer = word_seg(answer, vocab)
        questions.append(question)
        answers.append(answer)
    vocab_char = set(content)
    vocab_char_list = list(vocab_char)
    print('count:', len(questions), len(answers))
    vocab_dict = vocab
    #vocab = set(vocab.keys())
    vocab_dict_list = list(vocab_dict.items())
    vocab_dict_list.sort(key = lambda item:item[1], reverse=True)
    most_frequent_vocab = [item[0] for item in vocab_dict_list[:MAX_VOCAB_SIZE]]
    vocab = set(most_frequent_vocab)
    vocab = vocab | vocab_char

    to_be_removed = set()
    for c in vocab:
        if c != c.strip():
            to_be_removed.add(c)
    vocab = vocab - to_be_removed
    #print(vocab)
    print('vocab size:', len(vocab))
    vocab_dict_list = [item[0]+':'+str(item[1]) for item in vocab_dict_list]
    resegment_sentences(questions, vocab)
    resegment_sentences(answers, vocab)

    vocab = list(vocab)

    # Let's use separate vocab files for Q and A for now..
    save_list(os.path.join(OUT_DIR, VOCAB_FILE_NAME+'.'+QUES_ID), DEFAULT_VOCAB + vocab)
    save_list(os.path.join(OUT_DIR, VOCAB_FILE_NAME+'.'+ANS_ID), DEFAULT_VOCAB + vocab)
    save_list(os.path.join(OUT_DIR, VOCAB_FILE_NAME+'.'+'char-wise'), DEFAULT_VOCAB + vocab_char_list)
    save_list(os.path.join(OUT_DIR, VOCAB_FILE_NAME+'.'+'dict'), vocab_dict_list)
    

    questions_train, questions_test, answers_train, answers_test = train_test_split(questions, answers, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42, shuffle=True)
    questions_train, questions_dev, answers_train, answers_dev = train_test_split(questions_train, answers_train, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=42, shuffle=True)

    save_list(os.path.join(OUT_DIR, TRAIN_FILE_NAME+'.'+QUES_ID), questions_train)
    save_list(os.path.join(OUT_DIR, TRAIN_FILE_NAME+'.'+ANS_ID), answers_train)
    save_list(os.path.join(OUT_DIR, TEST_FILE_NAME+'.'+QUES_ID), questions_test)
    save_list(os.path.join(OUT_DIR, TEST_FILE_NAME+'.'+ANS_ID), answers_test)
    save_list(os.path.join(OUT_DIR, DEV_FILE_NAME+'.'+QUES_ID), questions_dev)
    save_list(os.path.join(OUT_DIR, DEV_FILE_NAME+'.'+ANS_ID), answers_dev)

def main():
    process(DATASET_PATH)

if __name__ == '__main__':
    main()
    