# NAME: sweep_tree_analysis_noenc_nli_RLSPINN
# NUM RUNS: 5
# SWEEP PARAMETERS: {'semantic_classifier_keep_rate': ('skr', 'LIN', 0.5, 1.0), 'l2_lambda': ('l2', 'EXP', 1e-09, 1e-06), 'learning_rate': ('lr', 'EXP', 0.0001, 0.001), 'tracking_lstm_hidden_dim': ('tlhd', 'EXP', 8, 64), 'rl_weight': ('rlwt', 'EXP', 1.0, 8.0)}
# FIXED_PARAMETERS: {'eval_seq_length': '810', 'log_path': '/scratch/sb6065/logs/spinn', 'embedding_keep_rate': '1.0', 'training_data_path': '/home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl', 'mlp_dim': '1024', 'seq_length': '80', 'model_dim': '600', 'word_embedding_dim': '300', 'use_internal_parser': '', 'ckpt_path': '/scratch/sb6065/logs/spinn', 'eval_interval_steps': '1000', 'data_type': 'nli', 'batch_size': '32', 'encode': 'projection', 'embedding_data_path': '/home/sb6065/glove/glove.840B.300d.txt', 'use_tracking_in_composition': '', 'statistics_interval_steps': '100', 'learning_rate_decay_per_10k_steps': '1.0', 'num_mlp_layers': '1', 'model_type': 'RLSPINN', 'eval_data_path': '/home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl', 'transition_weight': '1.0', 'sample_interval_steps': '1000'}

SPINNMODEL="spinn.models.rl_classifier" SPINN_FLAGS=" --eval_seq_length 810 --eval_interval_steps 1000 --data_type nli --log_path /scratch/sb6065/logs/spinn --embedding_keep_rate 1.0 --learning_rate 0.000133977989283 --training_data_path /home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl --rl_weight 7.57223918161 --batch_size 32 --mlp_dim 1024 --statistics_interval_steps 100 --embedding_data_path /home/sb6065/glove/glove.840B.300d.txt --l2_lambda 1.74168286789e-08 --encode projection --semantic_classifier_keep_rate 0.641283323332 --use_tracking_in_composition  --tracking_lstm_hidden_dim 12 --model_type RLSPINN --model_dim 600 --seq_length 80 --num_mlp_layers 1 --learning_rate_decay_per_10k_steps 1.0 --word_embedding_dim 300 --use_internal_parser  --ckpt_path /scratch/sb6065/logs/spinn --sample_interval_steps 1000 --transition_weight 1.0 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl --experiment_name sweep_tree_analysis_noenc_nli_RLSPINN_0-skr0.64-l21.7e-08-lr0.00013-tlhd12-rlwt7.6 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl:/home/sb6065/ptb.jsonl --expanded_eval_only_mode --write_eval_report" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1

SPINNMODEL="spinn.models.rl_classifier" SPINN_FLAGS=" --eval_seq_length 810 --eval_interval_steps 1000 --data_type nli --log_path /scratch/sb6065/logs/spinn --embedding_keep_rate 1.0 --learning_rate 0.000351159711163 --training_data_path /home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl --rl_weight 5.22527712246 --batch_size 32 --mlp_dim 1024 --statistics_interval_steps 100 --embedding_data_path /home/sb6065/glove/glove.840B.300d.txt --l2_lambda 9.7164493447e-09 --encode projection --semantic_classifier_keep_rate 0.541510734567 --use_tracking_in_composition  --tracking_lstm_hidden_dim 21 --model_type RLSPINN --model_dim 600 --seq_length 80 --num_mlp_layers 1 --learning_rate_decay_per_10k_steps 1.0 --word_embedding_dim 300 --use_internal_parser  --ckpt_path /scratch/sb6065/logs/spinn --sample_interval_steps 1000 --transition_weight 1.0 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl --experiment_name sweep_tree_analysis_noenc_nli_RLSPINN_1-skr0.54-l29.7e-09-lr0.00035-tlhd21-rlwt5.2 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl:/home/sb6065/ptb.jsonl --expanded_eval_only_mode --write_eval_report" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1

SPINNMODEL="spinn.models.rl_classifier" SPINN_FLAGS=" --eval_seq_length 810 --eval_interval_steps 1000 --data_type nli --log_path /scratch/sb6065/logs/spinn --embedding_keep_rate 1.0 --learning_rate 0.000116890827196 --training_data_path /home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl --rl_weight 1.03865126008 --batch_size 32 --mlp_dim 1024 --statistics_interval_steps 100 --embedding_data_path /home/sb6065/glove/glove.840B.300d.txt --l2_lambda 2.07839570865e-07 --encode projection --semantic_classifier_keep_rate 0.763352036949 --use_tracking_in_composition  --tracking_lstm_hidden_dim 24 --model_type RLSPINN --model_dim 600 --seq_length 80 --num_mlp_layers 1 --learning_rate_decay_per_10k_steps 1.0 --word_embedding_dim 300 --use_internal_parser  --ckpt_path /scratch/sb6065/logs/spinn --sample_interval_steps 1000 --transition_weight 1.0 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl --experiment_name sweep_tree_analysis_noenc_nli_RLSPINN_2-skr0.76-l22.1e-07-lr0.00012-tlhd24-rlwt1 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl:/home/sb6065/ptb.jsonl --expanded_eval_only_mode --write_eval_report" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1

SPINNMODEL="spinn.models.rl_classifier" SPINN_FLAGS=" --eval_seq_length 810 --eval_interval_steps 1000 --data_type nli --log_path /scratch/sb6065/logs/spinn --embedding_keep_rate 1.0 --learning_rate 0.000162401356609 --training_data_path /home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl --rl_weight 2.4147338547 --batch_size 32 --mlp_dim 1024 --statistics_interval_steps 100 --embedding_data_path /home/sb6065/glove/glove.840B.300d.txt --l2_lambda 1.70630919083e-07 --encode projection --semantic_classifier_keep_rate 0.712440125443 --use_tracking_in_composition  --tracking_lstm_hidden_dim 29 --model_type RLSPINN --model_dim 600 --seq_length 80 --num_mlp_layers 1 --learning_rate_decay_per_10k_steps 1.0 --word_embedding_dim 300 --use_internal_parser  --ckpt_path /scratch/sb6065/logs/spinn --sample_interval_steps 1000 --transition_weight 1.0 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl --experiment_name sweep_tree_analysis_noenc_nli_RLSPINN_3-skr0.71-l21.7e-07-lr0.00016-tlhd29-rlwt2.4 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl:/home/sb6065/ptb.jsonl --expanded_eval_only_mode --write_eval_report" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1

SPINNMODEL="spinn.models.rl_classifier" SPINN_FLAGS=" --eval_seq_length 810 --eval_interval_steps 1000 --data_type nli --log_path /scratch/sb6065/logs/spinn --embedding_keep_rate 1.0 --learning_rate 0.000474065983648 --training_data_path /home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl --rl_weight 4.44160441387 --batch_size 32 --mlp_dim 1024 --statistics_interval_steps 100 --embedding_data_path /home/sb6065/glove/glove.840B.300d.txt --l2_lambda 5.16616991175e-08 --encode projection --semantic_classifier_keep_rate 0.885317195555 --use_tracking_in_composition  --tracking_lstm_hidden_dim 40 --model_type RLSPINN --model_dim 600 --seq_length 80 --num_mlp_layers 1 --learning_rate_decay_per_10k_steps 1.0 --word_embedding_dim 300 --use_internal_parser  --ckpt_path /scratch/sb6065/logs/spinn --sample_interval_steps 1000 --transition_weight 1.0 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl --experiment_name sweep_tree_analysis_noenc_nli_RLSPINN_4-skr0.89-l25.2e-08-lr0.00047-tlhd40-rlwt4.4 --eval_data_path /home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl:/home/sb6065/ptb.jsonl --expanded_eval_only_mode --write_eval_report" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1