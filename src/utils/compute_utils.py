from src.utils.string_utils import NE_CHAR
import numpy as np


def compute_string_kernel(s1, s2):
    s1 = s1.replace(NE_CHAR, '$NE$')
    s2 = s2.replace(NE_CHAR, '$NE$')

    s1, s2 = s1.lower(), s2.lower()
    
    h, h2 = {}, {}
    for i in range(len(s2) - 5):
        h2[s2[i:i+6]] = True

    result = 0
    for i in range(len(s1) - 5):
        if s1[i:i+6] in h2 and s1[i:i+6] not in h:
            result += 1
            h[s1[i:i+6]] = True

    return result


def normalize_string_kernel(string_kernel):
    string_kernel = string_kernel.astype(np.float32)
    string_kernel[string_kernel == 0] = 1
    dp = np.array([string_kernel[i, i] for i in range(string_kernel.shape[0])])
    dp = np.reshape(dp, (dp.shape[0], 1))

    numitor = np.sqrt(np.matmul(dp, dp.transpose()))
    result = string_kernel / numitor
    
    return result