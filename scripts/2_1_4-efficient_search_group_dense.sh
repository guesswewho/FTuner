#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..
echo $1
cd ${PROJECT_ROOT}/ops/dense
source ${PROJECT_ROOT}/environ/activate_dev.sh
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1000}
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_group_dynT --group $1

mkdir -p 2_1-saved_artifacts
cd 2_1-saved_artifacts
cp ../*.csv .
