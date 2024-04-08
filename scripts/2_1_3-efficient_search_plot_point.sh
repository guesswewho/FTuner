#!/bin/bash -e

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/..

cd ${PROJECT_ROOT}/ops/dense
source ${PROJECT_ROOT}/environ/activate_dev.sh
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS:-1000}
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_5_dynT > point_score_5.out 2>&1
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_24_dynT > point_score_24.out 2>&1
AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_43_dynT > point_score_43.out 2>&1
#AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_62_dynT > point_score_62.out 2>&1
#AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_81_dynT > point_score_81.out 2>&1
#AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_100_dynT > point_score_100.out 2>&1
#AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_119_dynT > point_score_119.out 2>&1
#AUTO_SCHED_NTRIALS=${AUTO_SCHED_NTRIALS} pytest -s test_efficient_search.py::test_train_128_dynT > point_score_128.out 2>&1
mkdir -p 2_1-saved_artifacts
cd 2_1-saved_artifacts
cp ../*.csv .
