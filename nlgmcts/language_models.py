from nltk.corpus import shakespeare
from nltk.lm import Laplace
from nltk.lm.preprocessing import flatten
from nltk.util import everygrams
import re


class ShakespeareCharLanguageModel:
    def __init__(self, n=3):
        tokens = []
        for book in shakespeare.fileids():
            elt = shakespeare.xml(book)
            iterator = elt.getiterator()
            for node in iterator:
                lines = node.findall("LINE")
                for line in lines:
                    line_tokens = list(str(line.text))
                    line_tokens.insert(0, "<L>")
                    line_tokens.append("</L>")
                    tokens.append(line_tokens)
        t = (everygrams(x, max_len=n) for x in tokens)
        v = flatten(tokens)
        lm = Laplace(order=n)  # add-one smoothing
        lm.fit(t, v)

        self._n = n
        self._lm = lm
        self._tokenize_pattern = re.compile(r'(<L>)|(</L>)')

    def tokenize(self, text):
        if text is None: return None
        split = self._tokenize_pattern.split(text)
        tokens = []
        for s in split:
            if s != '' and s is not None:
                if s == "<L>" or s == "</L>":
                    tokens.append(s)
                else:
                    tokens.extend(list(s))
        return tokens

    def generate(self, num_chars=1, text_seed=None, random_seed=None):
        tokenized = self.tokenize(text_seed)
        generated = self._lm.generate(num_words=num_chars, text_seed=tokenized, random_seed=random_seed)
        return ''.join(generated)

    def perplexity(self, text):
        tokenized = self.tokenize(text)
        train = (everygrams(tokenized, max_len=self._n))
        return self._lm.perplexity(train)

    def entropy(self, text):
        tokenized = self.tokenize(text)
        train = (everygrams(tokenized, max_len=self._n))
        return self._lm.entropy(train)

    def vocab(self):
        return [w for w in self._lm.vocab]

    def score(self, char, context=None):
        tokenized = self.tokenize(context)
        return self._lm.score(char, tokenized)

    def vocab_scores(self, context=None):
        all = []
        tokenized = self.tokenize(context)
        for v in self._lm.vocab:
            all.append((v, self._lm.score(v, tokenized)))
        return reversed(sorted(all, key=lambda k: k[1]))

    def top_n_vocab(self, n, context=None):
        top_n = []
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<UNK>":
                v = next(vocab_scores)
            top_n.append(v[0])
        return top_n

    def top_n_vocab_with_weights(self, n, context=None):
        top_n = ([], [])
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<UNK>":
                v = next(vocab_scores)
            top_n[0].append(v[0])
            top_n[1].append(v[1])
        return top_n[0], self._normalize(top_n[1])

    def _normalize(self, probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    def order(self):
        return self._n
