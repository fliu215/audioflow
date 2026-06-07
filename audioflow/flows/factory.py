from .schedules import Uniform, LogitNormal
from .matchers import FlowMatcher


def get_flow(configs):
    t_schedule = get_schedule(configs["schedule"])
    fm = get_matcher(configs["matcher"], t_schedule)
    return fm


def get_schedule(configs):
    name = configs["name"]
    
    if name == "uniform":
        return Uniform()

    elif name == "logitnormal":
        return LogitNormal(configs["mu"], configs["sigma"])

    else:
        raise ValueError(name)


def get_matcher(configs, t_schedule):

    name = configs["name"]

    if name == "linear":
        return FlowMatcher(t_schedule)

    else:
        raise ValueError(name)