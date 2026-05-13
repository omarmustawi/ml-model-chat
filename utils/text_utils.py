import re
import numpy as np
import pandas as pd

def normalize_text(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        x = " ".join([str(i) for i in x if i is not None])

    try:
        if pd.isna(x):
            return ""
    except:
        pass

    return re.sub(r"\s+", " ", str(x).strip().lower())