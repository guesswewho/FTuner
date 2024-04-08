"""
pytest Extra Options
"""

def pytest_addoption(parser):
    parser.addoption('--sched_log_fname', action="store", type=str)
    parser.addoption('--N', action="store", type=int, default=16)
    parser.addoption('--C', action="store", type=int, default=128)
    parser.addoption('--H', action="store", type=int, default=640)
    parser.addoption('--W', action="store", type=int, default=640)
    parser.addoption('--K', action="store", type=int, default=128)
    parser.addoption('--R', action="store", type=int, default=7)
    parser.addoption('--S', action="store", type=int, default=7)
