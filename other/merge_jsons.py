import json

from tqdm import tqdm

data = {}
for i in tqdm(range(1000)):
    i = str(i).zfill(3)
    file = f"/mnt/raid5/mo/ff_aligned_unaligned/aligned/{i}_coordinates.txt"

    with open(file, 'r') as f:
        add_data = json.load(f)

    data.update(add_data)

with open("/mnt/raid5/mo/ff_aligned_unaligned/aligned/all_coordinates.txt", 'w') as outfile:
    json.dump(data, outfile)
