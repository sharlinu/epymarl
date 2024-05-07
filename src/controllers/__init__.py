REGISTRY = {}

from .basic_controller import BasicMAC
from .cqmix_controller import CQMixMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["cqmix_mac"] = CQMixMAC