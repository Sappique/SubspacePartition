# Research Log

## Background
The idea of this project is to use the Subspace Partition method to partition a LLMs activations into interpretable subspaces and then look at how the subspace are connected between layers to construct a circuit from them.

### Subspace Partition
The Subspace Partition method takes a LLM and a set of activations for one layer. It then finds subspaces that, when the activations are projected into them, have a minimum total distance of each activation to its nearest neighbor over all subspaces.

We know that LLMs encode more features than the dimension of their hidden representation by using superposition. We believe that features that can occur together (e.g. "the last token was 'red'" and "the answer is '4'") are (nearly) pairwise orthogonal while mutually exclusive features (e.g. "the last token was 'red'" and "the last token was 'blue'") are not. The method should group such pairwise orthogonal features into one subspace. Thus we expect that each subspace corresponds to a "meta-feature" (e.g. "what was the last token" or "what is the answer").

### Subspace Circuits
Once we have the subspaces for multiple layers of a LLM we do not need any additional activations to construct a circuit. That also means that our circuits are input independent. We can directly look at how attention heads and MLP layers transfer information from one subspace to another.

I believe that we need a transformer with a know circuit to develop and test the circuit finding method. Out current toy models all have trivial circuits: They have two subspace on the first layer and one on the last, so the circuit is the first two connected to the third one, there is no other possibility.

## Log

### during the AISS Incubator
We spend most time understanding the algorithm and the implementation. I modified it in various ways, converted the entry point to functions instead of command line arguments, and build tools for storing results and outputs.

We trained a toy model, the copy transformer, a two layer two head attention only transformer that was trained to repeat a sequence of unique tokens (e.g. given `<bos><a><b><c><d>` output `<a><b><d><d>`). We then looked at the attention pattern of the attention heads. It is know that such a transformer develops induction head, so we knew what patter we wanted to see. Training the model to repeat the sequence multiple time did not produce the desired pattern, but training one repetition did.

We then applied the subspace partition algorithm. Since we knew what the subspaces for induction heads should look like we knew what we wanted the output to be. We tried the algorithm with many combination of hyperparameters but the results never looked good. We then created a new hyperparameter to stop the algorithm after the desired number of subspaces is reached. That produced the desired subspaces: `out/subspace_partition/single_head_single_repetition_pattern_low_dim_bigger_vocabulary_again_forces_2_subspaces` and `out/subspace_partition/single_head_single_repetition_pattern_low_dim_masked_first_repetition_manually_enforce_two_subspaces`.
However, it felt a little like cheating and in a real application we would obviously no know how many subspace there are or how they should look like.

### after the AISS Incubator
I tried to train a slightly more complex toy model: A copy transformer that learns to repeat a sequence of unique bigrams (so each token can occur multiple times, but each pair of tokes occurs only once). Again I knew how the attention pattern should look like. I build tools for generating n-grams and training the models. The models I trained learned the task, but did not have the desired attention pattern.

### 2026-03-30
I implemented loading of models form the save format and architecture used in https://github.com/lacoco-lab/decompiling_transformers. I loaded the two models that performed the best on the unique bigram copy task and looked at their attention patterns. They had the desired pattern, but also redundant (seemingly) attention heads.