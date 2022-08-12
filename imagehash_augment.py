import cv2
import os
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
# from __future__ import absolute_import
from imagehash import crop_resistant_hash, ImageMultiHash
from functools import partial
import functools

def resize_with_aspect(image, width=None, height=None, inter=cv2.INTER_AREA, scale=False):
    (h, w) = image.shape[:2]

    if not scale and h < height and w < width:
        return image

    if width is None or height is None:
        return image
    if h > w:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # return Image.fromarray(img)

def crop_resistant_hash_from_cv():
    class CropResistantHash():
        def compare(self, hash1, hash2):
            assert isinstance(hash1, ImageMultiHash)
            assert isinstance(hash2, ImageMultiHash)
            return hash1 - hash2

        def compute(self, cv2_img):
            img = convert_from_cv2_to_image(cv2_img)
            return crop_resistant_hash(img)

    return CropResistantHash()

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
    elif name == 'crop_resistant_hash':
        func = crop_resistant_hash_from_cv()
    return func


def get_similar_augmented_images(func_name: str, query_path: str, augmentation: callable):
    func = select_func(func_name)
    query_image = cv2.imread(query_path)
    hash_query = func.compute(query_image)
    dist = []

    other_images = augmentation(query_image)
    other_images.append(query_image)
    for img in other_images:
        hash = func.compute(img)
        dist.append((func.compare(hash_query, hash), img))

    dist.sort(key=lambda x: x[0])
    # for idx, (d, img) in enumerate(dist):
    #     print(f'{d}')
    #     cv2.imshow(f'{d}', resize_with_aspect(img, width=320))
    #     if 'q' == cv2.waitKey(0):
    #         cv2.destroyAllWindows()

    return dist


def collage_images(similar_images: List[Tuple[float, np.ndarray]], tartget_path: str, col: int, row: int , width: int, height: int, scale=False):
    image_collage = np.zeros((height*row, width*col, 3))

    for i, (dist, image) in enumerate(similar_images):
        if i >= col*row:
            break

        image = resize_with_aspect(image, width=width, height=height, scale=False)
        cv2.putText(image, f'{dist}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        begin_width = (i % col) * width
        begin_height = (i // col) * height
        if image.shape[-1] != image_collage.shape[-1]:
            image = image[..., np.newaxis]
            image = np.tile(image, (1, 1, image_collage.shape[-1]))
        image_collage[begin_height:begin_height+image.shape[0], begin_width:begin_width+image.shape[1], :] = image

    distance = [dist for (dist, image) in similar_images]

    avg = sum(distance) / len(distance)
    var = sum([(d - avg)**2  for d in distance]) / len(distance)
    cv2.putText(image_collage, f'avg {avg:.2f} var {var:.2f}', (10, height//2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(tartget_path, image_collage)


def resize_func(image, interpolation=cv2.INTER_AREA):
    height, width = image.shape[:2]
    size = np.arange(0.25, 2.0, 0.25)

    power_size = [(int(w * width), int(h * height)) for w in size for h in size]
    result = [cv2.resize(image, dim, interpolation=interpolation) for dim in power_size]

    return result


def rotate_func(image):
    rotatation = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]

    result = [cv2.rotate(image, r) for r in rotatation]

    return result


def flip_func(image):
    rotatation = [0, 1, -1]

    result = [cv2.flip(image, r) for r in rotatation]

    return result


def translate_func(image):
    h, w = image.shape[:2]
    shift_x_ratio = np.arange(0.001, 0.1, 0.02)
    shift_y_ratio = np.arange(0.001, 0.1, 0.02)

    M = [np.float32([[1, 0, x*w], [0, 1, y*h]]) for x in shift_x_ratio for y in shift_y_ratio]
    shifted = [cv2.warpAffine(image, m, (w, h)) for m in M]

    return shifted

def gaussian_noise_func(image, mean=0, sigma=30, trials=5):
    noise = []
    for i in range(trials):
        n = np.zeros(image.shape, np.uint8)
        cv2.randn(n, mean, (sigma, sigma, sigma))
        noise.append(n)
    image_w_noise = [cv2.add(image, n) for n in noise]
    return image_w_noise

def pepper_salt_func(image, low=0, high=1, trials=5, threshold=0.1):
    image_w_noise = []
    for i in range(trials):
        n = np.zeros(image.shape[:2], np.float32)
        cv2.randu(n, low, high)

        im_sp = image.copy()
        im_sp [n < threshold] = 0
        im_sp [n > 1 - threshold] = 255
        image_w_noise.append(im_sp)

    return image_w_noise


def cutout_func(image, low=0, high=0.3, trials=5, threshold=0.2, fill=0):
    image_w_cut = []
    h, w = image.shape[:2]
    for i in range(trials):
        amount = random.randint(1, 10)
        img_cut = image.copy()
        for a in range(amount):
            sh = int(random.uniform(low, high) * h)
            sw = int(random.uniform(low, high) * w)
            x = random.randint(0, w-sw)
            y = random.randint(0, h-sh)
            img_cut[y:y+sh, x:x+sw] = fill
        image_w_cut.append(img_cut)

    return image_w_cut


def rotate_chain_gaussian_func(image):
    images = rotate_func(image)
    result = []
    for i in images:
        output = gaussian_noise_func(i)
        result.extend(output)
    return result


def flip_chain_gaussian_func(image):
    images = flip_func(image)
    result = []
    for i in images:
        output = gaussian_noise_func(i)
        result.extend(output)
    return result




def grayscale_func(image):
    result = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]

    return result


def crop_func(image, min=0.01, max=0.1, trials=1):
    image_w_cut = []
    h, w = image.shape[:2]
    for i in range(trials):
        sh = int(random.uniform(min, max) * h)
        sw = int(random.uniform(min, max) * w)
        img_cut = image[sh:h-sh, sw:w-sw]
        image_w_cut.append(img_cut)

    return image_w_cut


if __name__ == '__main__':
    root = 'D:\\Downloads\\NFT'
    output = 'D:\\Downloads\\NFT\\output'
    image_paths = os.listdir(root)
    width = 320
    height = 320
    col = 3
    row = 2

    all_func_name = ['AverageHash', 'BlockMeanHash', 'ColorMomentHash', 'MarrHildrethHash', 'PHash', 'RadialVarianceHash']
    all_func_name = ['crop_resistant_hash']
    # all_query_images = ['ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png', # chocolate man
    #                     '89357024335517944861413370507870940163174825132248692794845928227030815473665.png', # ape
    #                     '5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png', # KidzTokyo
    #                     ]
    all_query_images = ['03030.eth==ENS-- Ethereum Name Service.png'
                        ]
    # idx = image_paths.index('ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png') # chocolate man
    # idx = image_paths.index('89357024335517944861413370507870940163174825132248692794845928227030815473665.png') # ape
    # idx = image_paths.index('5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png') # KidzTokyo

    image_paths = [os.path.join(root, base) for base in image_paths if base.endswith('png') or base.endswith('jpg')]

    augmentations = [resize_func, rotate_func, flip_func, translate_func, pepper_salt_func, gaussian_noise_func, cutout_func, rotate_chain_gaussian_func, flip_chain_gaussian_func, grayscale_func]
    # augmentations = [grayscale_func]
    augmentations = [grayscale_func]
    augmentations = [partial(crop_func, min=0.05, max=0.2, trials=5)]
    for base in all_query_images:
        query_path = os.path.join(root, base)
        for aug in augmentations:
            for func_name in all_func_name:
                similar_image = get_similar_augmented_images(func_name, query_path, aug)
                target_basename = f'{func_name}_{aug.func.__name__[:-5]}_{base[:-4]}.jpg'
                target_path = os.path.join(output, target_basename)
                collage_images(similar_image, target_path, col=col, row=row, width=width, height=height)



