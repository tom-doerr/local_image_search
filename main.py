#!/usr/bin/env python3



from docarray import Document, DocumentArray
from clip_client import Client
import sys


DEFAULT_CLIP_URL = 'grpcs://demo-cas.jina.ai:2096' 

target = ' '.join(sys.argv[1:])


LOCAL_URL = 'grpc://localhost:51000'


try:
    c = Client(LOCAL_URL)
    vec = c.encode([target])
except ConnectionError:
    c = Client(DEFAULT_CLIP_URL)
    vec = c.encode([target])



import os
current_working_dir = os.getcwd()

CACHE_FILE = 'cache.csv'

import glob
import numpy as np

IMAGE_FILE_ENDINGS = ['jpg', 'png', 'jpeg', 'bmp', 'tiff', 'tif']

images = []
# all images in the folder
for file in glob.glob(current_working_dir + '/*'):
    if file.split('.')[-1] in IMAGE_FILE_ENDINGS:
        images.append(file)

# images = glob.glob(current_working_dir + "/*.jpg")

image_paths = images

from functools import lru_cache
import time

from joblib import Memory
memory = Memory(cachedir='./cache', verbose=0)

@memory.cache
def time_last_encode(filename):
    return time.time()

# @lru_cache(maxsize=None)
@memory.cache
def encode_image(image_path):
    da = [Document(uri=image_path)]
    da = c.encode(da, show_progress=True)
    doc_array = da[0]
    embedding_image = doc_array.embedding
    return embedding_image






results = []
for i, image_path in enumerate(image_paths):
    print(f'{i}/{len(image_paths)}')


    embedding_image = encode_image(image_path)

    import numpy as np
    norm_vec = np.linalg.norm(vec[0])
    normalized_vec = vec[0] / norm_vec

    norm_embedding_image = np.linalg.norm(embedding_image)
    normalized_embedding_image = embedding_image / norm_embedding_image

    logit_val = np.dot(normalized_vec, normalized_embedding_image)

    # r = da.find(query=vec, limit=9)
    results.append((image_path, logit_val))


results = sorted(results, key=lambda x: x[1], reverse=True)
best_results = results[:9]
worst_results = results[-9:]

print(f'Best results:')
for result in best_results:
    print(f'{result[1]:.3f} {result[0]}')

print(f'Worst results:')
for result in worst_results:
    print(f'{result[1]:.3f} {result[0]}')



