from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    num_simulations = 10000
    width = 3
    text_length = 25
    start_state = []

    eval_function = lambda text : -lm.perplexity(text)

    mcts = LanguageModelMCTS(lm, width, text_length, eval_function)
    state = start_state

    print("beginning search...")
    while len(state) < text_length:
        state = mcts.search(state, num_simulations)
        print(state)

    generated_text = ''.join(state)
    print("generated text: %s" % generated_text)
