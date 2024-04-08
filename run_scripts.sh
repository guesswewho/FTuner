./scripts/2_1-experiment_dynamic_dense.sh > dietcode_dense_fine_grained.out 2>&1
./scripts/2_2-experiment_dynamic_batch_matmul_nt.sh > dietcode_batchmatmul.out 2>&1
./scripts/2_4-experiment_bert.sh > dietcode_bert_base.out 2>&1
./scripts/2_6-experiment_bert_large.sh > dietcode_bert_large.out 2>&1

source launch_bounds_V100.sh
./scripts/2_1_1-efficient_search_dense.sh > Ftuner_dense_fine_grained.out 2>&1
./scripts/2_2_1-efficient_search_batch_matmul_nt.sh > Ftuner_batchmatmul.out 2>&1
./scripts/2_4_1-efficient_search_bert.sh > Ftuner_bert_base.out 2>&1
./scripts/2_5_1-efficient_search_gpt2.sh > Ftuner_gpt2.out 2>&1
./scripts/2_6_1-efficient_search_bert_large.sh > Ftuner_bert_large.out 2>&1