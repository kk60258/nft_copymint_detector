import cv2
import os
import random
from typing import List, Tuple
import numpy as np
import functools
import pathlib
from functools import partial
import mongo
from PIL import Image
from tqdm import tqdm

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


def rotate_func(image, rotation):
    # rotation cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE
    result = cv2.rotate(image, rotation)

    return result, f'{rotation}'


def flip_func(image, rotation: int=0):
    # rotation = [0, 1, -1]

    result = cv2.flip(image, rotation)

    return result, f'{rotation}'


def translate_func(image, range=0):
    h, w = image.shape[:2]
    shift_x_ratio = random.uniform(-range, range)
    shift_y_ratio = random.uniform(-range, range)

    M = np.float32([[1, 0, shift_x_ratio*w], [0, 1, shift_y_ratio*h]])
    shifted = cv2.warpAffine(image, M, (w, h))

    return shifted, f'range{range}'

def gaussian_noise_func(image, mean=0, sigma=100, trials=1):
    noise = []
    for i in range(trials):
        n = np.zeros(image.shape, np.uint8)
        cv2.randn(n, mean, (sigma, sigma, sigma))
        noise.append(n)
    image_w_noise = [cv2.add(image, n) for n in noise]
    return image_w_noise[0], f'sigma{sigma}_mean{mean}'


def contrast_brightness_func(img, contrast: float=1.0, brightness: int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    # contrast_pick = 1 if contrast == 1 else random.uniform(0, contrast)
    # brightness_pick = random.uniform(-brightness, brightness)

    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness), f'contrast{contrast}_brightness{brightness}'

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

def crop_func(image, min=0.01, max=0.1, trials=1):
    image_w_cut = []
    h, w = image.shape[:2]
    for i in range(trials):
        amount = random.randint(1, 1)
        # img_cut = image.copy()
        for a in range(amount):
            sh = int(random.uniform(min, max) * h)
            sw = int(random.uniform(min, max) * w)
            img_cut = image[sh:h-sh, sw:w-sw]
        image_w_cut.append(img_cut)

    return image_w_cut[0], f'{min}to{max}'

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


def imread(query_path):
    stream = open(query_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    query_image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    # query_image = cv2.imread(query_path)

    if query_image is None:
        # try gif
        gif = cv2.VideoCapture(query_path)
        ret,frame = gif.read() # ret=True if it finds a frame else False. Since your gif contains only one frame, the next read() will give you ret=False
        query_image = Image.fromarray(frame)

        query_image = np.asarray(query_image)
    return query_image

def get_mongo_filenames():
    mongo_address = "mongodb://localhost:27017/"
    mongo_db = 'imagehash'
    mongo_collection = 'hash_compute_filter3_0704'

    mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()
    existed_docs = mongo_helper.dump_image_hash()
    filter_names = [doc.file_name for doc in existed_docs]
    return filter_names


# def bg_color_change_func(image):
    # # Fill the black background with white color
    # #cv2.floodFill(image, None, seedPoint=(0, 0), newVal=(0, 0, 255), loDiff=(2, 2, 2), upDiff=(2, 2, 2))  # Not working!
    # image = image[:, :, :3]
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # rgb to hsv color space
    #
    # s_ch = hsv_img[:, :, 1]  # Get the saturation channel
    #
    # thesh = cv2.threshold(s_ch, 0, 255, cv2.THRESH_BINARY)[1]  # Apply threshold - pixels above 5 are going to be 255, other are zeros.
    # thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))  # Apply opening morphological operation for removing artifacts.
    #
    # cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1, upDiff=1)  # Fill the background in thesh with the value 128 (pixel in the foreground stays 0.
    #
    # image[thesh == 128] = (0, 0, 255)  # Set all the pixels where thesh=128 to red.
def bg_color_change_func(img, replace, seed=(0, 0), loDiff=10, upDiff=10):
    img = img[:, :, :3]
    img = cv2.resize(img, (640, 640))
    copyimg = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    # mask = np.zeros([h, w], np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur_img = cv2.medianBlur(gray, 5)
    blur_img = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)

    thresh = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    canvas = mask[1:h, 1:w]
    # canvas = mask
    cv2.drawContours(canvas, contours, -1, (255,255,255), -1)
    cv2.imshow('contours', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # copyimg[mask==255] = replace
    cv2.floodFill(copyimg, mask, seed, replace, (loDiff, loDiff, loDiff), (upDiff, upDiff, upDiff), cv2.FLOODFILL_FIXED_RANGE)
    return copyimg, f'{replace}'


def gamma_correction_func(src, gamma=1):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table), f'{gamma}'


def hue_func(img, range):
    # extract alpha channel
    if img.shape[-1] == 4:
        alpha = img[:,:,3]

    # extract bgr channels
    bgr = img[:,:,0:3]

    # convert to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    diff_color = random.randint(-range, range)

    # modify hue channel by adding difference and modulo 180
    hnew = np.mod(h + diff_color, 180).astype(np.uint8)

    # recombine channels
    hsv_new = cv2.merge([hnew, s, v])

    # convert back to bgr
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    if img.shape[-1] == 4:
        img_new = cv2.merge([bgr_new, alpha])
    else:
        img_new = bgr_new
    return img_new, f'{range}'

if __name__ == '__main__':
    # root = 'D:\\Downloads\\NFT'
    # output = 'D:\\Downloads\\NFT\\attack'
    root = 'D:\\github\\detect_copymint\\temp_0704'
    # output = 'D:\\github\\detect_copymint\\temp_0704_attack_test'
    output = 'D:\\github\\detect_copymint\\temp_0810_attack'
    # check_existed = 'D:\\github\\detect_copymint\\temp_0704_attack\\flip-0'
    # check_existed = os.listdir(check_existed)
    #
    large_n_file_path = ''#'D:\\github\\detect_copymint\\temp_0730_attack\\large_n.csv'

    image_paths = os.listdir(root)
    # print(f'existed {len(check_existed)}, image_paths {len(image_paths)}')
    # image_paths = [n for n in image_paths if n not in check_existed]
    # print(f'existed {len(check_existed)}, image_paths {len(image_paths)}')


    all_func_name = ['AverageHash', 'BlockMeanHash', 'ColorMomentHash', 'MarrHildrethHash', 'PHash', 'RadialVarianceHash']
    # all_query_images = ['ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png', # chocolate man
    #                     '89357024335517944861413370507870940163174825132248692794845928227030815473665.png', # ape
    #                     '5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png', # KidzTokyo
    #                     ]


    # all_query_images = ['ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png', # chocolate man
    #                     '89357024335517944861413370507870940163174825132248692794845928227030815473665.png', # ape
    #                     ]


    # idx = image_paths.index('ByyNjyyxyYTL8SBL1nwiV1P9TD6NtbuNi27Smf7mGvAR.png') # chocolate man
    # idx = image_paths.index('89357024335517944861413370507870940163174825132248692794845928227030815473665.png') # ape
    # idx = image_paths.index('5ZDUDumvoU3nh38cF5JJx3hWwUgbdwSEW2gBYxGEkSF2.png') # KidzTokyo

    image_paths = [base for base in image_paths if base.endswith('png') or base.endswith('jpg')]
    if large_n_file_path:
        large_n_files = open(large_n_file_path, 'r').readlines()
        large_n_files = [f.strip()[:-4] for f in large_n_files]
        image_paths = [p for p in image_paths if os.path.basename(p)[:-4] in large_n_files]

    filenames_mongo = get_mongo_filenames()
    image_paths_mongo = [base for base in image_paths if base in filenames_mongo]

    image_paths_mongo = ['Goon #6676==Goons of Balatroon.png']


    print(f'mongo {len(image_paths_mongo)}, dir {len(image_paths)}')

    # augmentations = [resize_func, rotate_func, flip_func, translate_func, pepper_salt_func, gaussian_noise_func, cutout_func, rotate_chain_gaussian_func, flip_chain_gaussian_func, grayscale_func]
    augmentations = [partial(gaussian_noise_func, sigma=30, mean=0), partial(gaussian_noise_func, sigma=100, mean=0),
                     partial(crop_func, min=0.05, max=0.1), partial(crop_func, min=0.1, max=0.2), partial(crop_func, min=0.2, max=0.25),
                     partial(translate_func, range=0.1), partial(translate_func, range=0.2),
                     partial(flip_func, rotation=0), partial(flip_func, rotation=1), partial(flip_func, rotation=-1),
                     partial(rotate_func, rotation=cv2.ROTATE_90_COUNTERCLOCKWISE), partial(rotate_func, rotation=cv2.ROTATE_180), partial(rotate_func, rotation=cv2.ROTATE_90_CLOCKWISE),
                     partial(contrast_brightness_func, contrast=1, brightness=40), partial(contrast_brightness_func, contrast=1, brightness=-40),
                     partial(contrast_brightness_func, contrast=3, brightness=0), partial(contrast_brightness_func, contrast=0.7, brightness=0),]
    augmentations = [partial(bg_color_change_func, replace=(0,0,255))]

    # augmentations = [partial(crop_func, min=0.01, max=0.01), partial(crop_func, min=0.03, max=0.03), partial(crop_func, min=0.05, max=0.05), partial(crop_func, min=0.07, max=0.07), partial(crop_func, min=0.1, max=0.1)]
    augmentations = [partial(hue_func, range=20), partial(hue_func, range=60), partial(hue_func, range=120)]
    augmentations += [partial(gamma_correction_func, gamma=0.5), partial(gamma_correction_func, gamma=2)]
    augmentations = [partial(crop_func, min=0.05, max=0.05), partial(crop_func, min=0.1, max=0.1),
                     partial(crop_func, min=0.15, max=0.15)]
    # augmentations = [partial(crop_func, min=0.0, max=0.0)]
    print(f'aug {len(augmentations)}')
    # augmentations = [grayscale_func]
    # for base in tqdm(all_query_images):
    for num, base in enumerate(tqdm(image_paths_mongo)):
        query_path = os.path.join(root, base)
        img = imread(query_path)
        for aug in augmentations:
            img_aug, parameter = aug(img)
            target_basename = f'{base[:-4]}.jpg'
            output_attack = os.path.join(output, f'{aug.func.__name__[:-5]}-{parameter}')
            pathlib.Path(output_attack).mkdir(parents=True, exist_ok=True)
            target_path = os.path.join(output_attack, target_basename)
            cv2.imwrite(target_path, img_aug)
        # if num > 10:
        #     break



