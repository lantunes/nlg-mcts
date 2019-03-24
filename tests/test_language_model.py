import unittest
from nlgmcts import ShakespeareCharLanguageModel


class TestLanguageModel(unittest.TestCase):

    def test_shakespeare_tokenize(self):
        lm = ShakespeareCharLanguageModel(n=3)

        tokenized = lm.tokenize(None)
        self.assertIsNone(tokenized)

        tokenized = lm.tokenize("")
        self.assertEqual([], tokenized)

        tokenized = lm.tokenize("The cat in the hat.")
        self.assertEqual(
            ['T', 'h', 'e', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'h', 'a', 't', '.'], tokenized)

        tokenized = lm.tokenize("<L>The cat in the hat.</L>")
        self.assertEqual(
            ['<L>', 'T', 'h', 'e', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'h', 'a', 't', '.', '</L>'], tokenized)

        tokenized = lm.tokenize("A world <L>apart.</L>")
        self.assertEqual(['A', ' ', 'w', 'o', 'r', 'l', 'd', ' ', '<L>', 'a', 'p', 'a', 'r', 't', '.', '</L>'], tokenized)

    def test_shakespeare_generate(self):
        lm = ShakespeareCharLanguageModel(n=3)

        generated = lm.generate(num_chars=5, text_seed="ab", random_seed=1)
        self.assertEqual("hold ", generated)

    def test_shakespeare_perplexity(self):
        lm = ShakespeareCharLanguageModel(n=3)

        line = "A dog and a cat"
        self.assertAlmostEqual(13.296, lm.perplexity(line), places=3)

        line = "A dog and a ggg"
        self.assertAlmostEqual(20.414, lm.perplexity(line), places=3)

        line = "A dog and a tac"
        self.assertAlmostEqual(15.579, lm.perplexity(line), places=3)

        line = "Adfsd ew xczv sq"
        self.assertAlmostEqual(122.858, lm.perplexity(line), places=3)

        line = "To be or not to be"
        self.assertAlmostEqual(9.099, lm.perplexity(line), places=3)

        line = "<L>To be, or not to be</L>"
        self.assertAlmostEqual(10.785, lm.perplexity(line), places=3)

        line = "<L>To be, or not to be, that is the question:</L>"
        self.assertAlmostEqual(9.985, lm.perplexity(line), places=3)

    def test_shakespeare_entropy(self):
        lm = ShakespeareCharLanguageModel(n=3)

        line = "A dog and a cat"
        self.assertAlmostEqual(3.733, lm.entropy(line), places=3)

        line = "A dog and a ggg"
        self.assertAlmostEqual(4.351, lm.entropy(line), places=3)

        line = "A dog and a tac"
        self.assertAlmostEqual(3.962, lm.entropy(line), places=3)

        line = "Adfsd ew xczv sq"
        self.assertAlmostEqual(6.941, lm.entropy(line), places=3)

        line = "To be or not to be"
        self.assertAlmostEqual(3.186, lm.entropy(line), places=3)

        line = "<L>To be, or not to be</L>"
        self.assertAlmostEqual(3.431, lm.entropy(line), places=3)

        line = "<L>To be, or not to be, that is the question:</L>"
        self.assertAlmostEqual(3.320, lm.entropy(line), places=3)

    def test_vocab(self):
        lm = ShakespeareCharLanguageModel(n=3)
        self.assertEqual(['<L>', 'N', 'a', 'y', ',', ' ', 'b', 'u', 't', 'h', 'i', 's', 'd', 'o', 'g', 'e', 'f',
                          'r', 'n', 'l', "'", '</L>', 'O', 'w', 'm', ':', 'T', 'H', 'v', 'k', 'p', 'M', 'c', 'U',
                          'W', 'A', '.', 'L', 'I', 'R', 'G', 'F', ';', '-', 'C', 'D', 'P', '!', 'Y', '?', 'E',
                          'q', 'K', 'x', 'B', 'z', 'S', 'V', 'J', 'j', 'Q', '&', 'Z', '<UNK>'], lm.vocab())

        self.assertEqual(['<L>', 'N', 'a', 'y', ',', ' ', 'b', 'u', 't', 'h', 'i', 's', 'd', 'o', 'g', 'e', 'f',
                          'r', 'n', 'l', "'", '</L>', 'O', 'w', 'm', ':', 'T', 'H', 'v', 'k', 'p', 'M', 'c', 'U',
                          'W', 'A', '.', 'L', 'I', 'R', 'G', 'F', ';', '-', 'C', 'D', 'P', '!', 'Y', '?', 'E',
                          'q', 'K', 'x', 'B', 'z', 'S', 'V', 'J', 'j', 'Q', '&', 'Z'], lm.vocab(with_unk=False))

    def test_score(self):
        lm = ShakespeareCharLanguageModel(n=3)

        score = lm.score("u", context="q")
        self.assertAlmostEqual(0.90, score, places=2)

        score = lm.score("l", context="ab")
        self.assertAlmostEqual(0.30, score, places=2)

    def test_vocab_scores(self):
        lm = ShakespeareCharLanguageModel(n=3)

        scores = lm.vocab_scores("ab")

        v = next(scores)
        self.assertEqual("l", v[0])
        self.assertAlmostEqual(0.3016, v[1], places=4)

        v = next(scores)
        self.assertEqual("o", v[0])
        self.assertAlmostEqual(0.2597, v[1], places=4)

        v = next(scores)
        self.assertEqual("s", v[0])
        self.assertAlmostEqual(0.0645, v[1], places=4)

        v = next(scores)
        self.assertEqual("i", v[0])
        self.assertAlmostEqual(0.0645, v[1], places=4)

        v = next(scores)
        self.assertEqual("u", v[0])
        self.assertAlmostEqual(0.0468, v[1], places=4)

        v = next(scores)
        self.assertEqual("e", v[0])
        self.assertAlmostEqual(0.0355, v[1], places=4)

        v = next(scores)
        self.assertEqual("r", v[0])
        self.assertAlmostEqual(0.0339, v[1], places=4)

        v = next(scores)
        self.assertEqual("a", v[0])
        self.assertAlmostEqual(0.0274, v[1], places=4)

        v = next(scores)
        self.assertEqual("b", v[0])
        self.assertAlmostEqual(0.0226, v[1], places=4)

        v = next(scores)
        self.assertEqual("y", v[0])
        self.assertAlmostEqual(0.0210, v[1], places=4)

        v = next(scores)
        self.assertEqual("h", v[0])
        self.assertAlmostEqual(0.0129, v[1], places=4)

        v = next(scores)
        self.assertEqual(",", v[0])
        self.assertAlmostEqual(0.0097, v[1], places=4)

        v = next(scores)
        self.assertEqual("j", v[0])
        self.assertAlmostEqual(0.0081, v[1], places=4)

        v = next(scores)
        self.assertEqual("</L>", v[0])
        self.assertAlmostEqual(0.0048, v[1], places=4)

        v = next(scores)
        self.assertEqual(" ", v[0])
        self.assertAlmostEqual(0.0048, v[1], places=4)

        v = next(scores)
        self.assertEqual("-", v[0])
        self.assertAlmostEqual(0.0032, v[1], places=4)

        v = next(scores)
        self.assertEqual(":", v[0])
        self.assertAlmostEqual(0.0032, v[1], places=4)

        v = next(scores)
        self.assertEqual("<UNK>", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("Z", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("&", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("Q", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("J", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("V", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("S", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("z", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("B", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("x", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("K", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("q", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("E", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("?", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("Y", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("!", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("P", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("D", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("C", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual(";", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("F", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("G", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("R", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("I", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("L", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual(".", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("A", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("W", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("U", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("c", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("M", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("p", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("k", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("v", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("H", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("T", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("m", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("w", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("O", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("'", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("n", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("f", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("g", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("d", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("t", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("N", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

        v = next(scores)
        self.assertEqual("<L>", v[0])
        self.assertAlmostEqual(0.0016, v[1], places=4)

    def test_top_n_vocab(self):
        lm = ShakespeareCharLanguageModel(n=3)

        top_n = lm.top_n_vocab(3, "ab")
        self.assertEqual(["l", "o", "s"], top_n)

        top_n = lm.top_n_vocab(3, "e a g")
        self.assertEqual(['Z', '&', 'Q'], top_n)

    def test_top_n_vocab_with_weights(self):
        lm = ShakespeareCharLanguageModel(n=3)

        top_n = lm.top_n_vocab_with_weights(3, "ab")
        self.assertEqual(["l", "o", "s"], top_n[0])
        self.assertEqual(3, len(top_n[1]))
        self.assertAlmostEqual(0.4820, top_n[1][0], places=4)
        self.assertAlmostEqual(0.4149, top_n[1][1], places=4)
        self.assertAlmostEqual(0.1031, top_n[1][2], places=4)

        top_n = lm.top_n_vocab_with_weights(3, "e a g")
        self.assertEqual(["Z", "&", "Q"], top_n[0])
        self.assertEqual(3, len(top_n[1]))
        self.assertAlmostEqual(0.3333, top_n[1][0], places=4)
        self.assertAlmostEqual(0.3333, top_n[1][1], places=4)
        self.assertAlmostEqual(0.3333, top_n[1][2], places=4)

    def test_order(self):
        lm = ShakespeareCharLanguageModel(n=2)
        self.assertEqual(2, lm.order())
