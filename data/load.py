import lxml.etree
import re
import nltk
from collections import Counter
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
from collections import defaultdict
import os
import pickle


class TedDataset(Dataset):

    def __init__(self, ted_data, labels_ted=None, vocabulary=None, min_frequency=10):

        self.has_output = False

        if vocabulary is not None:
            self.vocabulary = vocabulary
        else:
            self.vocabulary = get_docs_vocabulary(
                ted_data, min_frequency=min_frequency
            )

        self.indexer = get_word_indexer(self.vocabulary)
        self.data = {
            'input': tokens2indices(ted_data, self.indexer),
        }
        if labels_ted:
            self.data['output'] = labels2indices(labels_ted)
            self.has_output = True

        self.padding_index = self.vocabulary.index('__pad__')

    def __len__(self):
        return len(self.data['input'])

    def __getitem__(self, index):
        if self.has_output:
            return {'input': self.data['input'][index],
                    'output': self.data['output'][index]}
        else:
            return {'input': self.data['input'][index]}

    def collate_fn(self, data):

        batch = {}

        input_data = [x['input'] for x in data]
        max_length = max([len(sent) for sent in input_data])
        input_batch = torch.ones(len(input_data), max_length) * self.padding_index
        for i, sent in enumerate(input_data):
            input_batch[i, :len(sent)] = torch.LongTensor(sent)

        batch['input'] = input_batch

        if self.has_output:
            output_data = [x['output'] for x in data]
            batch['output'] = torch.LongTensor(output_data)

        return batch


def load_ted_data(xml_file):

    pkl_file = '{}.pkl'.format(os.path.basename(xml_file).split('.')[0])

    if os.path.isfile(pkl_file):
        tokenized_text, labels = pickle.load(open(pkl_file, 'rb'))
    else:
        input_text, keywords = load_ted_xml(xml_file)
        input_text = preprocess_ted_data(input_text)
        tokenized_text = tokenize_sentences(input_text)
        labels = keywords2labels(keywords)
        pickle.dump((tokenized_text, labels), open(pkl_file, 'wb'))

    return tokenized_text, labels


def split_dataset(tokenized_sentences):
    train = tokenized_sentences[:-500]
    dev = tokenized_sentences[-500:-250]
    test = tokenized_sentences[-250:]
    return train, dev, test


def get_docs_vocabulary(tokens, min_frequency=5):
    word_count = get_word_counts(tokens)
    vocabulary = [x[0] for x in word_count.items() if x[1] >= min_frequency]
    vocabulary = ['__unk__', '__pad__'] + vocabulary

    return vocabulary


def tokens2indices(tokens, indexer):
    return recursive_map(tokens, lambda x: indexer[x])


def indices2tokens(indices, vocabulary):
    return recursive_map(indices, lambda x: vocabulary[x])


def get_word_indexer(vocabulary):
    indexer = defaultdict(lambda : vocabulary.index('__unk__'))
    indexer.update({vocabulary[idx]: idx for idx in range(len(vocabulary))})
    return indexer


def load_ted_xml(xml_file):
    doc = lxml.etree.parse(open(xml_file, 'r'))
    docs_raw_text = doc.xpath('//content/text()')
    keywords = doc.xpath('//keywords/text()')
    return docs_raw_text, keywords


def preprocess_ted_data(documents_raw_text):

    documents_ted = []
    for input_text in documents_raw_text:
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

        sentences_ted = list(filter(lambda x: x != ' ', sentences_ted))

        documents_ted.append('\n'.join(sentences_ted))

    return documents_ted


def keywords2labels(strings_keywords):
    labels = []
    for sent in strings_keywords:
        keywords = set(sent.lower().split(', '))
        if {'technology', 'entertainment', 'design'} in keywords:
            labels.append('TED')
        elif {'technology', 'entertainment'} in keywords:
            labels.append('TEo')
        elif {'technology', 'design'} in keywords:
            labels.append('ToD')
        elif {'entertainment', 'design'} in keywords:
            labels.append('oED')
        elif 'technology' in keywords:
            labels.append('Too')
        elif 'entertainment' in keywords:
            labels.append('oEo')
        elif 'design' in keywords:
            labels.append('ooD')
        else:
            labels.append('ooo')

    return labels


def labels2indices(labels):
    label_mapper = {
        'TED': 0,
        'TEo': 1,
        'ToD': 2,
        'oED': 3,
        'Too': 4,
        'oEo': 5,
        'ooD': 6,
        'ooo': 7
    }

    return recursive_map(labels, lambda x: label_mapper[x])


def tokenize_sentences(sentence_strings):
    return recursive_map(sentence_strings, lambda x: nltk.word_tokenize(x))


def get_word_counts(tokenized_sentences):
    return Counter(list(itertools.chain.from_iterable(tokenized_sentences)))


def recursive_map(l, f, dtype=list):
    if isinstance(l, dtype):
        return list(map(lambda x: recursive_map(x, f), l))
    return f(l)


if __name__ == '__main__':
    tokens_ted, labels = load_ted_data('ted_en-20160408.xml')
    tokens_train, tokens_dev, tokens_test = split_dataset(tokens_ted)
    labels_train, labels_dev, labels_test = split_dataset(labels)
    train_dataset = TedDataset(tokens_train, labels_train, min_frequency=10)
    dev_dataset = TedDataset(tokens_dev, labels_dev, vocabulary=train_dataset.vocabulary)
    test_dataset = TedDataset(tokens_test, labels_test, vocabulary=train_dataset.vocabulary)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=3,
        num_workers=4
    )

    for batch in train_dataloader:
        print(batch)
        break
