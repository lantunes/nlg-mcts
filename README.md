MCTS for NLG
============

TODO
- create a trigram character-level language model
- create a language-model informed MCTS class 
- compare the language model to the MCTS and MCTS+LM classes, in terms of how often they are able to generate valid 3-letter, 4-letter, 5-letter, etc. words
-- examine the effect of varying the number of MCTS simulations
- to evaluate the efficiency of MCTS, generate n-letter words randomly as a baseline, and see how many attempts are required before valid words are generated
-- this will tell us if the number of MCTS simulations is reasonable
- consider also trying the single continuous search MCTS approach, and see how that compares
-- the idea is to stop the search as soon as the first valid word is found

- we can use the output (i.e. the generated text) in an Expert Iteration process that improves the language model that makes the suggestions in the tree

- how does this approach compare with Beam Search? 
