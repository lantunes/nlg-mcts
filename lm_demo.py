from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    best = None
    for _ in range(5000):
        generated = lm.generate(num_chars=50, text_seed="<L>")
        generated = "<L>" + generated  # the generated text does not contain the text seed prefix
        perplexity = lm.perplexity(generated)
        print((generated, perplexity))
        if best is None or perplexity < best[1]:
            best = (generated, perplexity)
        print("best: ", best)
