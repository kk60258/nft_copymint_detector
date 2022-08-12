import numpy as np
import cv2
from PIL import Image
import os
from imagehash import crop_resistant_hash, ImageMultiHash, ImageHash


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


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    np_img = np.array(img)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def crop_resistant_hash_from_cv():
    class CropResistantHash():
        def compare(self, hash1, hash2):
            if not isinstance(hash1, ImageMultiHash):
                assert isinstance(hash1, list) and isinstance(hash1[0], np.ndarray)
                hash1 = ImageMultiHash([ImageHash(h) for h in hash1])

            if not isinstance(hash2, ImageMultiHash):
                assert isinstance(hash2, list) and isinstance(hash2[0], np.ndarray)
                hash2 = ImageMultiHash([ImageHash(h) for h in hash2])

            return hash1 - hash2

        def compute(self, cv2_img):
            img = convert_from_cv2_to_image(cv2_img)

            hash_func = None#BlockMeanHashForImageHash()
            return crop_resistant_hash(img)            #0730
            # return crop_resistant_hash(img, hash_func=hash_func, limit_segments=4, segment_threshold=128,
            #                            min_segment_size=8100, segmentation_image_size=300) # 0808
            # return crop_resistant_hash(img, hash_func=hash_func, limit_segments=4, segment_threshold=128,
            #                            min_segment_size=2500, segmentation_image_size=300)            #0809

    return CropResistantHash()


class BlockMeanHashForImageHash(object):
    def __init__(self):
        self.func = select_func('BlockMeanHash')

    def __call__(self, *args, **kwargs):
        img = args[0]
        np_img = convert_from_image_to_cv2(img)
        hash = self.func.compute(np_img)
        return ImageHash(hash)


def select_func(name) -> cv2.img_hash.ImgHashBase:
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


def clean_filename(s):
    if not s:
        return ''
    bad_chars = '\\/:*?\"<>|,'
    for c in bad_chars:
        s = s.replace(c, '_')
    return s


def compute_hash_distance(query_hash: str, target_hash, *, func_name: str=None, func: cv2.img_hash.ImgHashBase=None):
    if func_name:
        func = select_func(func_name)
    dist = func.compare(query_hash, target_hash)
    return dist