import cv2
import os
import random
from typing import List, Tuple
import numpy as np


# query = 'ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png'
#
# path = os.path.join(root, query)
# image = cv2.imread(path)
# func = cv2.img_hash.AverageHash_create()
# hash = cv2.img_hash.averageHash(image)
#
# cv2.imshow('query', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# def dist_hamming(s1: str, s2: str) -> int:
#     if len(s1) != len(s2):
#         return float('inf')
#     return sum([c1 != c2 for (c1, c2) in zip(s1, s2)])
#
#
# print(hash)


def resize_with_aspect(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None or height is None:
        return image
    if h > w:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def select_func(name):
    if name == 'AverageHash':
        func = cv2.img_hash.AverageHash_create()
    elif name == 'BlockMeanHash':
        func = cv2.img_hash.BlockMeanHash_create()
    elif name == 'ColorMomentHash':
        func = cv2.img_hash.ColorMomentHash_create()
    elif name == 'MarrHildrethHash':
        func = cv2.img_hash.MarrHildrethHash_create()
    elif name == 'PHash':
        func = cv2.img_hash.PHash_create()
    elif name == 'RadialVarianceHash':
        func = cv2.img_hash.RadialVarianceHash_create()
    return func


def get_similar_images(func_name: str, query_path: str, other_image_paths: List[str]):
    func = select_func(func_name)
    query_image = cv2.imread(query_path)
    hash_query = func.compute(query_image)
    print(f'query hash {hash_query}')
    # cv2.imshow('query', resize_with_aspect(query_image, width=320))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dist = []

    for p in other_image_paths:
        img = cv2.imread(p)
        hash = func.compute(img)
        dist.append((func.compare(hash_query, hash), p))

    dist.sort(key=lambda x: x[0])
    # dist.insert(0, (0, query_path))
    # for idx, (d, p) in enumerate(dist):
    #     print(f'{d} {p}')
    #     img = cv2.imread(p)
    #     cv2.imshow(f'{d}', resize_with_aspect(img, width=320))
    #     if 'q' == cv2.waitKey(0):
    #         cv2.destroyAllWindows()

    return dist


def collage_images(similar_image_paths: List[Tuple[float, str]], tartget_path: str, col: int, row: int , width: int, height: int):
    image_collage = np.zeros((height*row, width*col, 3))

    for i, (dist, path) in enumerate(similar_image_paths):
        if i >= col*row:
            break
        image = cv2.imread(path)
        image = resize_with_aspect(image, width=width, height=height)
        cv2.putText(image, f'{dist}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        begin_width = (i % col) * width
        begin_height = (i // col) * height

        image_collage[begin_height:begin_height+image.shape[0], begin_width:begin_width+image.shape[1], :] = image

    cv2.imwrite(tartget_path, image_collage)

if __name__ == '__main__':
    root = 'D:\\Downloads\\NFT'
    output = 'D:\\Downloads\\NFT\\output_by_images'
    image_paths = os.listdir(root)
    width = 320
    height = 320
    col = 3
    row = 2

    all_func_name = ['AverageHash', 'BlockMeanHash', 'ColorMomentHash', 'MarrHildrethHash', 'PHash', 'RadialVarianceHash']
    all_query_images = ['ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png', # chocolate man
                        '89357024335517944861413370507870940163174825132248692794845928227030815473665.png', # ape
                        '5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png', # KidzTokyo
                        ]
    # idx = image_paths.index('ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png') # chocolate man
    # idx = image_paths.index('89357024335517944861413370507870940163174825132248692794845928227030815473665.png') # ape
    # idx = image_paths.index('5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png') # KidzTokyo

    image_paths = [os.path.join(root, base) for base in image_paths if base.endswith('png') or base.endswith('jpg')]

    for base in all_query_images:
        query_path = os.path.join(root, base)
        for func_name in all_func_name:
            similar_image_paths = get_similar_images(func_name, query_path, image_paths)
            target_basename = f'{func_name}_{base[:-4]}.jpg'
            target_path = os.path.join(output, target_basename)
            collage_images(similar_image_paths, target_path, col=col, row=row, width=width, height=height)



