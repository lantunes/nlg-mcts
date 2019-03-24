from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    num_simulations = 250000
    text_length = 50
    start_state = ["<L>"]

    eval_function = lambda text_state: -lm.perplexity(''.join(text_state))

    mcts = TextMCTS(lm.vocab(with_unk=False), text_length, eval_function, c=10)
    state = start_state

    print("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    print("generated text: %s (score: %s)" % (generated_text, str(best[1])))
