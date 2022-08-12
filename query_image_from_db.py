import cv2
import os
import random
from typing import List, Tuple
import numpy as np
import mongo
from collections import defaultdict
from tqdm import tqdm
from PIL import Image





from util import select_func, crop_resistant_hash_from_cv, imread, clean_filename, compute_hash_distance


def flip_func(image):
    rotatation = [0, 1, -1]
    result = [cv2.flip(image, r) for r in rotatation]
    return result

def rotate_func(image):
    rotatation = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]

    result = [cv2.rotate(image, r) for r in rotatation]

    return result


def compute_hash(query_path: str, *, func_name: str=None, func: cv2.img_hash.ImgHashBase=None, rotate=False):
    if func_name:
        func = select_func(func_name)
    query_image = imread(query_path)
    total_images = [query_image]
    hashes = []
    if rotate:
        flip_images = flip_func(query_image)
        rotate_images = rotate_func(query_image)
        total_images.extend(flip_images)
        total_images.extend(rotate_images)

    hashes = [func.compute(q) for q in total_images]
    return hashes








if __name__ == '__main__':
    hash_rotate = False
    # root = 'D:\\Downloads\\NFT\\attack'
    # output = 'D:\\Downloads\\NFT\\attack_result.csv'
    # root = 'D:\\github\\detect_copymint\\temp_0715_attack'
    # output = 'D:\\github\\detect_copymint\\temp_0715_attack\\attack_result.csv'

    root = 'D:\\github\\detect_copymint\\temp_0803_attack'
    output = 'D:\\github\\detect_copymint\\temp_0803_attack\\attack_result.csv'

    large_n_file_path = 'D:\\github\\detect_copymint\\temp_0730_attack\\large_nn.csv'
    attacks = os.listdir(root)
    image_paths = [os.path.join(root, f) for root, dirs, files in os.walk(root) for f in files if f.endswith('.jpg') or f.endswith('.png')]

    if large_n_file_path:
        large_n_files = open(large_n_file_path, 'r').readlines()
        large_n_files = [f.strip()[:-4] for f in large_n_files]
        image_paths = [p for p in image_paths if os.path.basename(p)[:-4] in large_n_files]
    # image_paths = ['D:\\github\\detect_copymint\\temp_0730_attack\\crop-0.1to0.1\\Goon #6676==Goons of Balatroon.jpg']
    # image_paths = ['D:\\github\\detect_copymint\\temp_0704\\Goon #6676==Goons of Balatroon.png']
    # all_func_name = ['AverageHash', 'BlockMeanHash', 'ColorMomentHash', 'MarrHildrethHash', 'PHash', 'RadialVarianceHash']
    # all_func_name = ['PHash']
    all_func_name = ['crop_resistant_hash']
    # all_query_images = ['03030.eth==ENS-- Ethereum Name Service.jpg',
    #                     ]
    # image_paths = [os.path.join(root, f) for root, dirs, files in os.walk(root) for f in files if f in all_query_images]

    mongo_address = "mongodb://localhost:27017/"
    mongo_db = 'imagehash'
    # mongo_collection = 'hash_compute_filter3_0704'
    # mongo_collection = 'hash_compute_test'
    # mongo_collection = 'hash_crop_resistant_0809_large_nn'
    mongo_collection = 'hash_crop_resistant_0810_large_nn'

    mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()
    image_hash_all = []
    if hash_rotate:
        types = ['base', 'flip0', 'flip1', 'flip2', 'rotate0', 'rotate1', 'rotate2']
    else:
        types = ['base']
    print('compute hash')
    for query_path in tqdm(image_paths):
        query_base_name = os.path.basename(query_path)
        attack_name = os.path.basename(os.path.dirname(query_path))
        image_hash_doc = mongo.DocumentImageHash(file_name=query_base_name)

        for func_name in all_func_name:
            func = select_func(func_name)
            hashes = compute_hash(query_path, func=func, rotate=hash_rotate)
            # print(f'{query_base_name} with {attack_name}: {hash}')

            for type, hash in zip(types, hashes):
                image_hash_doc.add_hash(hash=mongo.FieldHash(hash=hash, version=1, method=func_name, type=type))

        image_hash_all.append((query_base_name, attack_name, image_hash_doc))



    docs = mongo_helper.dump_image_hash()
    all_dist_result = defaultdict(list)
    print('\nfind db hash')
    for query_base_name, attack_name, query_image_hash_doc in tqdm(image_hash_all):
        min_n = float('inf')
        min_n_dist = float('inf')
        for func_name in all_func_name:
            func = select_func(func_name)
            dist_result = []
            for doc in docs:
                field_hash = doc.get_hash(func_name, 'base')
                min_dist = float('inf')

                if doc.file_name[:-4] == query_base_name[:-4]:
                    print(f'{query_base_name} start hash compare')

                for type in types:
                    query_hash = query_image_hash_doc.get_hash(func_name, type)
                    dist = compute_hash_distance(query_hash.hash, field_hash.hash, func=func)
                    min_dist = min(min_dist, dist)
                    break
                if doc.file_name[:-4] == query_base_name[:-4]:
                    print(f'{query_base_name}: min_dist {min_dist}')

                if min_dist == float('inf'):
                    print(f'cannot find {doc.file_name}')
                dist_result.append((doc.file_name, min_dist))

            dist_result = sorted(dist_result, key=lambda x: x[1])
            # with open(os.path.join(os.path.dirname(output), 'dist_goon6776.csv'), 'w', encoding='UTF-8') as f:
            #     for dist in dist_result:
            #         f.write(f'{dist[1]}, {dist[0]}\n')
            find_list = [(i,v[1]) for i, v in enumerate(dist_result) if v[0][:-4] == query_base_name[:-4]]
            if len(find_list) == 0:
                break

            n, dist = find_list[0]
            if 0 <= n < min_n:
                min_n = n
                min_n_dist = dist
            # print(dist_result)
        all_dist_result[query_base_name].append((attack_name, min_n, min_n_dist))

    to_text = []
    total = len(docs)
    for query_base_name, l in all_dist_result.items():
        for attack_name, min_n, dist in l:
            s = f'{clean_filename(query_base_name)},{attack_name},{min_n},{total},{dist}'
            # print(s)
            to_text.append(s)

    # exit()
    with open(output, 'w', encoding='UTF-8') as f:
        f.write('name,attack,min_n,total,dist\n')
        f.writelines('\n'.join(to_text))
