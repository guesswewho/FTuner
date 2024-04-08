#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

cd ${PROJECT_ROOT}/ops/batch_matmul
source ${PROJECT_ROOT}/environ/activate_dev.sh
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1000}
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_dynT_1

mkdir -p 2_2-saved_artifacts
cd 2_2-saved_artifacts
cp ../*.csv .
