from nlgmcts import *

if __name__ == '__main__':

    print("creating language model...")
    lm = ShakespeareCharLanguageModel(n=5)

    num_simulations = 100
    width = 3
    text_length = 50
    start_state = ["<L>"]

    eval_function = lambda text: 100 - lm.perplexity(text)

    mcts = LanguageModelMCTS(lm, width, text_length, eval_function, c=25)
    state = start_state

    print("beginning search...")
    while len(state) < text_length:
        state = mcts.search(state, num_simulations)
        print(state)

    generated_text = ''.join(state)
    print("generated text: %s (%s)" % (generated_text, str(lm.perplexity(generated_text))))
