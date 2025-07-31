### Intro
This repo contains code for the paper Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning

### Repo Structure

- `toy-model/` - code for toy setting
  - `model_train.py` - train a toy model of intra-group superposition
  - `model_analyze.py` - check orthogonality of obtained feature vectors
  - `train_search_merge.py` - run NDM for toy model
  
- `training/` - code for training R for LMs

  - `train_search.py` - main file to run NDM (merging-based) training
  - `model.py` - contains definition of matrix trainer
  
  - `train_search_split.py` - split-based NDM (discussed in Appendix)

  - `train_adversarial_recon.py` `train_adversarial_disc.py` - Minimax approach (discussed in Appendix)
  - others are self-explanatory
- `trainedRs` - each foler contains matrices trained in one experiment. It now contains 3 folders for the best R we obtained for GPT2 Small, Qwen2.5-1.5B and Gemma-2-2B respectively. Matrix weights, training log, training arguments, evaluation raw data are all contained. We also release partition trained for each layer post-MLP residual stream of GPT-2 Small.

- `evaluate` - code for evaluating trained Rs
  - `evaluate.py` - run GPT2 test suite for an experiment
  - `evaluate_conflict.py` - run subspace patching in knowledge conflict setting
  - others are self-explanatory
  
- `preimage` - code for making the APP
  - `app.py` - entry point of the APP, requires saved faiss index and input text
  - `page/`
    - `with_attribution.py` - the second page, requires saved preimages
  - `cache_act.py` - run model and save activations
  - `build_index.py` - load saved activations and project them into subspaces defined by matrices in a experiment, and build faiss index
  - `cache_attribution` - pre-compute attribution scores for limited number of preimages.
- `visualizations` - output of the pipeline in `preimage`


### Run Experiments

- Toy setting:  `model_train.py` -> `model_analyze.py` -> `train_search_merge.py` change hyperparameters inside each file.
- LM experiments: 
  1. Go to `training` folder, and `python train_search.py --exp_name [EXP_NAME]`
Change other arguments as you want in command line. Like mentioned, configurations we used can be found in `trainedRs/[EXP_NAME]/training_args.json`
   2. Go to `evaluate` folder, and `python evaluate.py --exp_name [EXP_NAME]` or `python evaluate_confict.py --exp_name [EXP_NAME]`
   3. Go to `preimage` folder. Run `cache_act.py --model_name [MODEL_NAME]` (this only needs to run once for each model). Run `build_index.py --exp_name [EXP_NAME]` to build and save faiss index. Run `cache_attribution.py  --exp_name [EXP_NAME]` (This step is not need if you don't need attribution scores, and it's kind of time consuming). All 3 steps in this pipeline provide `--override` option if you want to override already saved data. 
   4. Finally run `streamlit run app.py [EXP_NAME]` and check your browser.