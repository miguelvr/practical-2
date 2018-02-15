import lxml.etree
import re
import nltk
from collections import Counter
import itertools
from torch.utils.data import Dataset, DataLoader


def load_ted_data(xml_file):
    input_text = load_ted_xml(xml_file)
    input_text = preprocess_ted_data(input_text)
    tokenized_text = tokenize_sentences(input_text)

    return tokenized_text


def split_dataset(tokenized_sentences):
    train = tokenized_sentences[:-500]
    dev = tokenized_sentences[-500:-250]
    test = tokenized_sentences[-250:]
    return train, dev, test


def get_vocabulary(tokens, min_frequency=5):
    word_count = get_word_counts(tokens)
    vocabulary = [x[0] for x in word_count.items() if x[1] >= min_frequency]
    vocabulary = ['__unk__'] + vocabulary

    return vocabulary


def tokens2indices(tokens, indexer):
    return recursive_map(tokens, lambda x: indexer[x])


def indices2tokens(indices, vocabulary):
    return recursive_map(indices, lambda x: vocabulary[x])


def get_word_indexer(vocabulary):
    return {vocabulary[idx]: idx for idx in range(len(vocabulary))}


def load_ted_xml(xml_file):
    doc = lxml.etree.parse(open(xml_file, 'r'))
    input_text = '\n'.join(doc.xpath('//content/text()'))
    del doc

    return input_text


def preprocess_ted_data(input_text):
    input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

    sentences_strings_ted = []
    for line in input_text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings_ted.extend(
            sent.lower() for sent in m.groupdict()['postcolon'].split('.') if sent
        )

    sentences_ted = []
    for sent_str in sentences_strings_ted:
        sentences_ted.append(re.sub(r"[^a-z0-9]+", " ", sent_str.lower()))

    return sentences_ted


def tokenize_sentences(sentence_strings):
    return [nltk.word_tokenize(sent) for sent in sentence_strings]


def get_word_counts(tokenized_sentences):
    return Counter(list(itertools.chain.from_iterable(tokenized_sentences)))


def recursive_map(l, f, dtype=list):
    if isinstance(l, dtype):
        return list(map(lambda x: recursive_map(x, f), l))
    return f(l)


if __name__ == '__main__':
    tokens_ted = load_ted_data('ted_en-20160408.xml')
    tokens_train, tokens_dev, tokens_test = split_dataset(tokens_ted)
    vocab = get_vocabulary(tokens_train, min_frequency=10)
    indexer = get_word_indexer(vocab)

    indices = tokens2indices(tokens_train[:2], indexer)
    tokens = indices2tokens(indices, vocab)

    print(indices)
    print(tokens)
