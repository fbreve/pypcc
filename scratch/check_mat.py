from scipy.io import loadmat
import numpy as np

dir_path = r"C:\Users\fbrev\Documents\Acadêmico\Simulações\Matlab\Geral\ssl-book"
for i in [8, 9]:
    file_path = f"{dir_path}\\SSL,set={i},data.mat"
    try:
        mat = loadmat(file_path)
        print(f"=== set={i} ===")
        for k, v in mat.items():
            if not k.startswith("__"):
                if isinstance(v, np.ndarray):
                    print(f"Key: {k}, type: {type(v)}, shape: {v.shape}, dtype: {v.dtype}")
                else:
                    print(f"Key: {k}, type: {type(v)}")
    except Exception as e:
        print(f"Error loading set={i}: {e}")
