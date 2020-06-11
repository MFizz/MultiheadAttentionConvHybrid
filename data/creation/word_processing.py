import tensorflow as tf
import pandas as pd


def norm_sentence(sentence):
    """
    Returns the same sentence in lower case, without commas, words in parenthesis and add a stop symbol </s>.
    :param sentence: Input string.
    :return: Processed string.
    """
    if sentence is None:
        return None

    # all words to lower case:
    sentence = str.lower(sentence)

    # get rid of words in parenthesis
    while '(' in sentence:
        sentence = sentence[:sentence.find('(')] + sentence[sentence.find(')') + 2:]

    # get rid of punctuation
    sentence = ''.join(char for char in sentence if char not in '!"#$%&*+,-./:;<=>?@^_')

    # add stop token
    if sentence[len(sentence)-1] == ' ':
        sentence += u"</s>"
    else:
        sentence += u" </s>"

    # add start token
    sentence = u"<s> " + sentence

    return sentence


def get_context_words(w_id, window, sentence):
    """
    Gets context elements around target element given a window size.
    For example given window size 2, target element id 4 and input list [0,1,2,3,4,5,6,7,8,9,10] the return would be:
    [2,3,5,6]
    :param w_id: Id of target element.
    :param window: Window size to the left and window size to the right of the target element.
    :param sentence: List of elements.
    :return: List of context words.
    """
    # copy list so we don't delete in place
    new_sentence = list(sentence)
    del new_sentence[w_id]
    # get the window around the now deleted word
    context_words = new_sentence[max(0, w_id - window): min(len(new_sentence), w_id + window)]
    return context_words


def create_vocabulary(sentence_list):
    """
    Creates a unique word list out of a sentence list.
    :param sentence_list: Target sentence list.
    :return: Unique word list.
    """
    splits_list = [sentence.split(" ") for sentence in sentence_list]
    lowercase_splits_list = [[str.lower(word) for word in sentence] for sentence in splits_list]
    word_list = [word for split in lowercase_splits_list for word in split]
    return set(word_list)


def read_csv_label_sentences(csv_filename):
    """
    Reads csv file containing sentences, adds a stop symbol "</s>" to each sentence and adds it to a list. Also deletes
    words in parenthesis and commas as I have no idea how to handle them.
    :param csv_filename: Csv file containing one target sentence per row.
    :return: A list of target sentences with stop symbol added.
    """
    sentences = []
    # with tf.gfile.GFile(csv_filename, mode='rb') as labels_csv:
    cols = ['sentence']
    labels = pd.read_csv(FileWrap(csv_filename), delimiter=';', header=None, names=cols)

    sentences = [norm_sentence(row['sentence']) for _, row in labels.iterrows()]
    # for _, row in labels.iterrows():
    #     sentence = norm_sentence(row['sentence'])
    #     sentences.append(sentence)
    return sentences


def read_csv_labels(csv_filename):
    pass


class FileWrap:
    def __init__(self, path):
        self.file = open(path)
    def __iter__(self):
        self.file.readline().rstrip()
    def read(self, *args, **kwargs):
        return self.file.read()