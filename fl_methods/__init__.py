from .fedavg import FedAvg
from .fedprox import FedProx
from .scaffold import SCAFFOLD
# from .FedL2R import FedL2R
from .FedSR import FedSR

_method_class_map = {
    'fedavg': FedAvg,
    'fedprox': FedProx,
    # 'fedl2r': FedL2R,
    'feda': FedSR,
    'fedaga': FedSR,
    'scaffold': SCAFFOLD
}


def get_fl_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
