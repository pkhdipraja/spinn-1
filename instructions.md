## Instructions for SPINN 

The trained models that can match the performance in SPINN paper (SP-PI) can be accessed @swp20195 under `/project/kahardipraja/nyu-spinn/spinn-1`. The training took approximately 12 hours to converge.

#### Details for NYU SPINN implementation
  * This version of SPINN implementation is tested for Python 3.5.2 and PyTorch 1.0.1. Extra dependencies are listed under `python/requirements.txt`


#### Expected input format
  * This SPINN implementation requires NLI format (SNLI and MultiNLI), utilizing the binary parse `python/spinn/data/nli/load_nli_data.py`
  * It also supports other data type (MT, SST, Listops, etc.) For more information, see the --data_type flags.

#### Flags relevant for standard SPINN (full list of arguments under `python/spinn/models/base.py`)
  * `show_progress_bar`: by default show the progress bar, turn off when running experiments on HPC 
  * `experiment_name`: your desired experiment name
  * `load_experiment_name`: name of experiment to be loaded 
  * `data_type`: type of dataset (nli, mt, listops, etc.)
  * `log_path`: directory to write logs
  * `load_log_path`: directory to read logs from
  * `write_proto_to_log`: write logs in protocol buffer format
  * `ckpt_path`: directory to save/load checkpoint
  * `load_best`: if set, load 'best' checkpoint
  * `training_data_path`: path to your training data
  * `eval_data_path`: path to your evaluation data. Can evaluate multiple data separated with `:`. First argument should be devset
  * `seq_length`: maximum sequence length of training data
  * `allow_cropping`: if set, trim overly long training examples
  * `allow_eval_cropping`: if set, trim overly long evaluation examples
  * `eval_seq_length`: maximum sequence length of evaluation data
  * `embedding_data_path`: path to embeddings with GloVe format
  * `fine_tune_loaded_embeddings`: if set, backpropagate error signal to embeddings for update 
  * `model_type`: type of model (SPINN, RLSPINN, LMS, etc.)
  * `gpu`: if set use gpu, by default use cpu.
  * `model_dim`: set dimension of model
  * `word_embedding_dim`: set dimension of word embeddings 
  * `use_internal_parser`: if set use internal parse, by default use predicted parse 
  * `embedding_keep_rate`: use dropout on transformed embeddings and encoder RNN
  * `use_difference_feature`: use difference of premise and hypothesis as feature
  * `use_product_feature`: use product of premise and hypothesis as feature
  * `tracking_lstm_hidden_dim`: if set use tracker with specified dimension as described in SPINN paper, by default use no tracker
  * `tracking_ln`: if set use layer normalization in tracking
  * `transition_weight`: if set predict transitions, by default not predicting transitions
  * `lateral_tracking`: use previous tracker state as input for new state
  * `use_tracking_in_composition`: use tracking LSTM output as input for reduce function
  * `composition_ln`: use layer normalization in TreeLSTM composition
  * `predict_use_cell`: use cell output as feature for transition net
  * `reduce`: set composition function (TreeLSTM, TreeGRU, tanh, LMS)
  * `encode`: set preprocessing settings for embeddings. By default is projection as described in SPINN paper.
  * `encode_bidirectional`: if set, encode in both directions
  * `encode_num_layers`: RNN layers in encoding net
  * `mlp_dim`: dimension for SNLI classifier
  * `num_mlp_layers`: number of layers for SNLI classifier
  * `mlp_ln`: use layer normalization between MLP layers
  * `semantic_classifier_keep_rate`: dropout rate for SNLI classifier
  * `optimizer_type`: by default use SGD. There is also Adam optimizer
  * `training_steps`: set training steps
  * `batch_size`: set minibatch size
  * `learning_rate`: set learning rate
  * `learning_rate_decay_when_no_progress`: decay learning rate by this rate at every epoch when last epoch did not produce a new best result
  * `clipping_max_value`: set gradient clipping value
  * `l2_lambda`: set term for L2 regularization
  * `statistics_interval_steps`: log training statistics at this interval
  * `eval_interval_steps`: evaluate at this interval
  * `ckpt_interval_steps`: update checkpoint at this interval
  * `early_stopping_steps_to_wait`: if performance on devset doesn't improve after this many steps, stops training
  * `expanded_eval_only_mode`: if set, load checkpoint to do inference
  
#### Training
  * Here is a sample command that can be used for training (assuming current location is under `/python` directory):
```bash 
python3 -m spinn.models.supervised_classifier --data_type nli \
--training_data_path <path to training data> \
--eval_data_path <path to evaluation data> \
--embedding_data_path <path to embeddings> \
--word_embedding_dim 10 --model_dim 10 --model_type SPINN --gpu 0 \
--log_path <path to logs> --experiment_name spinn_sample
```


#### Inference
  * For inference, the sample command from above can be used with additional expanded_eval_only_mode flags.
```bash 
python3 -m spinn.models.supervised_classifier --data_type nli \
--training_data_path <path to training data> \
--eval_data_path <path to evaluation data> \
--embedding_data_path <path to embeddings> \
--word_embedding_dim 10 --model_dim 10 --model_type SPINN --gpu 0 \
--log_path <path to logs> --experiment_name spinn_sample \
--expanded_eval_only_mode
```

#### Example to reproduce SPINN paper with parameters 
```bash
python3 -m spinn.models.supervised_classifier --data_type nli \
--training_data_path ~/project/kahardipraja/.data/snli/snli_1.0/snli_1.0_train.jsonl \
--eval_data_path ~/project/kahardipraja/.data/snli/snli_1.0/snli_1.0_test.jsonl \
--embedding_data_path ~/project/kahardipraja/GloVe/glove.840B.300d.txt \
--word_embedding_dim 300 --model_dim 600 --model_type SPINN --gpu 0 \
--log_path ~/project/kahardipraja/nyu-spinn/spinn-1/logs/ \
--tracking_lstm_hidden_dim 61 --show_progress_bar --mlp_dim 1024 --num_mlp_layers 2 \ 
--l2_lambda 2e-5 --learning_rate 0.007 --optimizer_type Adam --seq_length 80 \
--eval_seq_length 810 --semantic_classifier_keep_rate 0.93 \
--embedding_keep_rate 0.92--experiment_name spinn_sample
```

#### Additional available models in the implementation 
* [RL-SPINN](https://arxiv.org/pdf/1611.09100.pdf): pretty similar to SPINN, but use reinforcement learning to train SPINN tracker to parse input sentence w/o external parsing data.
* [Maillard](https://arxiv.org/pdf/1705.09189.pdf): TreeLSTM applied to tree structure found by a fully differentiable natural language chart parser, eliminating the need for external parse.
* [Gumbel-Tree LSTM/ST-Gumbel](https://arxiv.org/pdf/1707.02786.pdf): can learn task-specific tree structures only from plain text data efficiently.
* [Lifted Matrix-Space](https://arxiv.org/pdf/1711.03602.pdf): to provide better scaling capability, use global transformation to map word embeddings to matrices, which then can be composed based on matrix-matrix multiplication. 


