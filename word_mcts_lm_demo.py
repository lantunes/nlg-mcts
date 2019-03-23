from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    num_simulations = 1000
    alphabet = lm.vocab()
    text_length = 25
    start_state = []

    eval_function = lambda text : -lm.perplexity(text)
    # eval_function = lambda text: -lm.entropy(text)

    mcts = WordMCTS(alphabet, text_length, eval_function)
    state = start_state

    while len(state) < text_length:
        state = mcts.search(state, num_simulations)
        print(state)

    generated_text = ''.join(state)
    print("generated text: %s" % generated_text)
