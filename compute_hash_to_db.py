import cv2
import os
import random
from typing import List, Tuple
import numpy as np
import mongo
from PIL import Image
from tqdm import tqdm
from imagehash import crop_resistant_hash, ImageMultiHash, ImageHash, dhash
from util import select_func, imread


def compute_hash(query_path: str, *, func_name: str=None, func: cv2.img_hash.ImgHashBase=None):
    if func_name:
        func = select_func(func_name)

    # too many weird filename
    # print(f'query {query_path}')
    query_image = imread(query_path)

    hash_query = func.compute(query_image)
    # cv2.imshow('q', query_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return hash_query
# D:\github\detect_copymint\test_0704\#1550json==GoblinAi.jpg
if __name__ == '__main__':
    dont_filter = True
    root = 'D:\\github\\detect_copymint\\temp_0704' # 'D:\\github\\detect_copymint\\temp_0803_attack\\crop-0.1to0.1'
    output = 'D:\\github\\detect_copymint\\temp_0811'
    large_n_file_path = ''#'D:\\github\\detect_copymint\\temp_0730_attack\\large_nn.csv'
    image_paths = os.listdir(root)

    # all_func_name = ['AverageHash', 'BlockMeanHash', 'ColorMomentHash', 'MarrHildrethHash', 'PHash', 'RadialVarianceHash']
    func_name = 'PHash'
    func_name = 'crop_resistant_hash'
    all_query_images = ['ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png', # chocolate man
                        '89357024335517944861413370507870940163174825132248692794845928227030815473665.png', # ape
                        '5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png', # KidzTokyo
                        ]


    image_paths = [base for base in image_paths if base.endswith('png') or base.endswith('jpg')]
    if large_n_file_path:
        large_n_files = open(large_n_file_path, 'r').readlines()
        large_n_files = [f.strip()[:-4] for f in large_n_files]
        image_paths = [p for p in image_paths if os.path.basename(p)[:-4] in large_n_files]
    print(f'total image {len(image_paths)}')

    mongo_address = "mongodb://localhost:27017/"
    mongo_db = 'imagehash'
    mongo_collection = 'hash_crop_resistant_0811'

    mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()
    image_hash_doc_all = {}
    existed_docs = mongo_helper.dump_image_hash()
    filter_hashes = [(doc.file_name, doc.get_hash(func_name, 'base').hash) for doc in existed_docs]
    filter_names = [doc.file_name for doc in existed_docs]
    skip_images = []
    if dont_filter:
        filter_hashes = []
        filter_names = []

    for idx, query_base_name in enumerate(tqdm(image_paths)):
        # if idx != 61:
        #     continue
        query_path = os.path.join(root, query_base_name)
        image_hash_doc = mongo.DocumentImageHash(file_name=query_base_name)

        func = select_func(func_name)
        hash = compute_hash(query_path, func=func)
        if not dont_filter and query_base_name in filter_names:
            print(f'skip====={query_base_name}')
            # skip_images.append(f'{query_base_name}')
        else:
            for name, existed in filter_hashes:
                # if np.array_equal(hash, existed):
                #     print(f'skip====={query_base_name}')
                #     skip_images.append((query_base_name, name))
                #     break
                if func.compare(existed, hash) <= 0:
                    print(f'skip====={query_base_name}')
                    skip_images.append(f'{query_base_name}, {name}')
                    break

            else:
                if not filter_hashes:
                    filter_hashes.append((query_base_name, hash))
                    filter_names.append(query_base_name)
                # print(f'{query_base_name} {hash}')

                image_hash_doc.add_hash(hash=mongo.FieldHash(hash=hash, version=1, method=func_name, type='base'))

                image_hash_doc_all.update({query_base_name: image_hash_doc})
                mongo_helper.upsert(image_hash_doc)
        # if len(image_hash_doc_all) > 10:
        #     break

    # docs = mongo_helper.dump_image_hash()
    # for doc in docs:
    #     print(f'{doc.to_doc()}')

    with open(os.path.join(output, 'skip.txt'), 'w', encoding='UTF-8') as f:
        f.writelines('skip,exist\n')
        f.writelines('\n'.join(skip_images))

    print('DONE')
    # docs = mongo_helper.dump_image_hash()
    #
    # for doc in docs:
    #     if doc.file_name in image_hash_doc_all:
    #         image_hash_doc = image_hash_doc_all[doc.file_name]
    #         # field_hash_dict = doc.__dict__
    #         # image_hash_dict = image_hash_doc.__dict__
    #
    #         valid_field_hash = doc.get_all_hashes_dict()
    #         valid_image_hash = image_hash_doc.get_all_hashes_dict()#[v for k, v in image_hash_dict.items() if isinstance(v, mongo.FieldHash)]
    #
    #         shared_items = [1 for k, v in valid_field_hash.items() if k in valid_image_hash and (v.hash == valid_image_hash[k].hash).all()]
    #         print(f'{doc.file_name} field length {len(valid_field_hash)}, image length {len(valid_image_hash)}, same length {len(shared_items)}')


