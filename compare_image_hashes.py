from util import imread, select_func
import os

if __name__ == '__main__':
    source = 'D:\\github\\detect_copymint\\temp_0810_attack\\crop-0.0to0.0'
    db = source

    query = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.jpg') or f.endswith('.png')]
    target = [os.path.join(db, f) for f in os.listdir(db) if f.endswith('.jpg') or f.endswith('.png')]
    func = select_func('crop_resistant_hash')

    query_hashes = []
    for q in query:
        img = imread(q)
        print(f'{os.path.basename(q)}')
        hash = func.compute(img)
        query_hashes.append((os.path.basename(q), hash))

    target_hashes = []
    if source == db:
        target_hashes = list(query_hashes)
    else:
        for t in target:
            img = imread(t)
            hash = func.compute(img)
            target_hashes.append((os.path.basename(t), hash))

    match = []
    for name1, hash1 in query_hashes:
        find = ''
        min_dist = float('inf')
        for name2, hash2 in target_hashes:
            distance = hash1 - hash2
            print(f'{name1} vs {name2}: {distance}')
            if distance < min_dist:
                min_dist = distance
                find = name2
        print(f'{name1} find {find}: {min_dist}')
