"""
pytest Extra Options
"""

def pytest_addoption(parser):
    parser.addoption('--micro_kernel', action="store", type=str)
    parser.addoption('--sched_log_fname', action="store", type=str)
    parser.addoption('-B', action="store", type=int, default=16)
    parser.addoption('-T', action="store", type=int, default=128)
    parser.addoption('-I', action="store", type=int, default=768)
    parser.addoption('-H', action="store", type=int, default=2304)
    parser.addoption('--group', action="store", type=int, default=128)
