from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    num_simulations = 1000
    width = 3
    text_length = 50
    start_state = ["<L>"]

    eval_function = lambda text: -lm.perplexity(text)

    mcts = LanguageModelMCTS(lm, width, text_length, eval_function)
    state = start_state

    print("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    print("generated text: %s (score: %s)" % (generated_text, str(best[1])))
