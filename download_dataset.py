import urllib.request
import os

os.makedirs("official_demos", exist_ok=True)
url = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/demo.hdf5"
print("Downloading...")
urllib.request.urlretrieve(url, "official_demos/lift_demos.hdf5")
print("Done!")