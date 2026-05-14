import multiprocessing
import time
import os

def worker_task(i):
    print(f"Worker {i} starting (PID: {os.getpid()})")
    time.sleep(5)
    print(f"Worker {i} finishing")

if __name__ == "__main__":
    print(f"Main process (PID: {os.getpid()})")
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(worker_task, range(8))
