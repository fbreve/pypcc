import time
from sklearn.datasets import fetch_openml
import socket

print("Setting socket timeout to 15s...")
socket.setdefaulttimeout(15)

print("Fetching 'letter' dataset...")
t0 = time.time()
try:
    ds = fetch_openml(name="letter", version="active", as_frame=False)
    print(f"Success! Data shape: {ds.data.shape}. Time taken: {time.time()-t0:.2f}s")
except Exception as e:
    print(f"Failed: {e}")
