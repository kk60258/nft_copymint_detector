import pymongo
from abc import ABC, abstractmethod
from bson.binary import Binary
import pickle
import numpy as np
from imagehash import ImageMultiHash

def cls_to_field(obj):
    if not hasattr(obj, "__dict__"):
        if isinstance(obj, np.ndarray):
            return ndarray_to_field(obj)
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("__"):
            continue
        element = []
        if isinstance(val, list):
            for item in val:
                element.append(cls_to_field(item))
        else:
            element = cls_to_field(val)
        result[key] = element
    return result


def ndarray_to_field(ndarray: np.ndarray):
    # convert numpy array to Binary, store record in mongodb
    s = pickle.dumps(ndarray, protocol=4)
    field = Binary(s, subtype=0x80)
    # print(f'{ndarray} => field type {field}')
    return field


def field_to_ndarray(field):
    # get field value from mongodb, convert Binary to numpy array
    if isinstance(field, list):
        ndarray_list = [pickle.loads(f) for f in field]
        # print(f'{field} => ndarray_list {ndarray_list}')
        return ndarray_list
    else:
        ndarray = pickle.loads(field)
        return ndarray


class Document(ABC):
    def __init__(self):
        self._id = None

    def to_doc(self):
        return cls_to_field(self)

    def __str__(self):
        return self.to_doc()


class Field(ABC):
    pass


class FieldHash(Field):
    def __init__(self, *, hash, version=1, method='average_hash', type='base', save=True):
        if save:
            if isinstance(hash, ImageMultiHash):
                self.hash = [h.hash for h in hash.segment_hashes]
            else:
                self.hash = hash
        else:
            self.hash = field_to_ndarray(hash)


        self.version = version
        assert 'hash' in method.lower()
        self.method = method
        self.type = type


class DocumentImageHash(Document):
    def __init__(self, *, file_name: str, hash: FieldHash = None):
        self.file_name = file_name
        self._id = file_name
        if hash:
            self.add_hash(hash)

    def add_hash(self, hash: FieldHash):
        setattr(self, f'{hash.method}_{hash.type}', hash)

    def get_hash(self, method, type):
        return getattr(self, f'{method}_{type}')

    def get_all_hashes_dict(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, FieldHash)}

class MongoHelper(object):
    def __init__(self, *, mongo_address, mongo_db, mongo_collection):

        self.mongo_address = mongo_address
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.client = None

    def connect(self):
        self.mongo_client = pymongo.MongoClient(self.mongo_address)
        self.db_client = self.mongo_client[self.mongo_db]
        self.collection_client = self.db_client[self.mongo_collection]

    def upsert(self, doc: Document):
        update = doc.to_doc()
        if '_id' in update:
            query = {'_id': update['_id']}
        else:
            query = update
        # print(f'query: {query}, update: {update}')
        self.collection_client.update_one(query, {"$set": update}, upsert=True)

    def dump_image_hash(self) -> DocumentImageHash:
        items = self.collection_client.find()
        docs = []
        for item in items:
            doc = DocumentImageHash(file_name=item['file_name'])
            for k, v in item.items():
                if not k.startswith('_') and 'hash' in k.lower():
                    # hash = DocumentHash(hash=v['hash'], version=v['version'], method=v['method'], type=v['type'])
                    hash = FieldHash(**v, save=False)
                    doc.add_hash(hash)
            docs.append(doc)
        return docs


if __name__ == '__main__':
    mongo_address = "mongodb://localhost:27017/"
    mongo_db = 'imagehash'
    mongo_collection = 'hash_test'

    mongo_helper = MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()

    image_hash_doc = DocumentImageHash(file_name='123.jpg')
    image_hash_doc.add_hash(hash=FieldHash(hash=[np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])], version=1, method='average_hash', type='base'))
    for k, v in image_hash_doc.__dict__.items():
        print(f'{k}')
    # for k in dir(image_hash_doc):
    #     print(f'{k}')
    mongo_helper.upsert(image_hash_doc)

    docs = mongo_helper.dump_image_hash()
    for doc in docs:
        print(f'{doc.to_doc()}')
