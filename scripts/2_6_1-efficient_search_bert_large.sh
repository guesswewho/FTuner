#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1500}
source ${PROJECT_ROOT}/environ/activate_dev.sh

cd ${PROJECT_ROOT}/ops/dense
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_BERT_LARGE_I1024_H4096
cp dietcode_autosched_dense_16xTx1024x4096.json saved_schedules_G4
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_BERT_LARGE_I1024_H1024
cp dietcode_autosched_dense_16xTx1024x1024.json saved_schedules_G4
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_BERT_LARGE_I4096_H1024
cp dietcode_autosched_dense_16xTx4096x1024.json saved_schedules_G4

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .

cd ${PROJECT_ROOT}/ops/batch_matmul
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_BERT_LARGE_dynT_1
cp dietcode_autosched_batch_matmul_nt_256xTx64xT.json saved_schedules_G4
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_BERT_LARGE_dynT_2
cp dietcode_autosched_batch_matmul_nt_256xTxTx64.json saved_schedules_G4

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .

cd ${PROJECT_ROOT}/networks/bert
# ./download_bert_large_uncased.sh
pytest -s test_efficient_search.py::test_infer_large_dynT

mkdir -p 2_4-saved_artifacts
cd 2_4-saved_artifacts
cp ../*.csv .
