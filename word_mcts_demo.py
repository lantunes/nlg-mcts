from nlgmcts import *
from nltk import *
from nltk.corpus import brown
from nltk.corpus import shakespeare
import string


if __name__ == '__main__':

    num_simulations = 10000
    alphabet = list(string.ascii_lowercase)
    word_length = 5
    start_state = ['t']

    # words = [w.lower() for w in brown.words()]
    all_words = []
    for book in shakespeare.fileids():
        all_words.extend(shakespeare.words(book))
    words = set([w.lower() for w in all_words])

    eval_function = lambda word: 1 if ''.join(word) in words else 0

    mcts = TextMCTS(alphabet, word_length, eval_function)
    state = start_state

    while len(state) < word_length:
        state = mcts.search(state, num_simulations)
        print(state)

    generated_word = ''.join(state)
    print("generated word: %s" % generated_word)
    print("is in corpus: %s" % (generated_word in words))
