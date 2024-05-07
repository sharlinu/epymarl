REGISTRY = {}

from .rnn_agent import RNNAgent
from .facmac_agent import FACMACRNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_facmac"] = FACMACRNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
