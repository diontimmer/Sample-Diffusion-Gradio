import os
from shared.util import get_recc_device, load_config_from_yaml

CONFIG = load_config_from_yaml()
REFRESH_SYMBOL = "\U0001f504"
MODEL_FOLDER = (
    CONFIG["MODEL_FOLDER"]
    if (os.getenv("SDGFOLDER") is None)
    else os.getenv("SDGFOLDER")
)
RECC_DEVICE = get_recc_device()
DEVICE_LIST = list(set(["cpu", RECC_DEVICE]))
