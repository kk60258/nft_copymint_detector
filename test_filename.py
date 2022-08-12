import os
import mongo

dir = 'D:\\github\\detect_copymint\\test_0708'
files = os.listdir(dir)
#
# for basename in files:
#     print(basename)

mongo_address = "mongodb://localhost:27017/"
mongo_db = 'imagehash'
# mongo_collection = 'hash_compute_filter3_0704'
mongo_collection = 'hash_compute_filter3_0704'

mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
mongo_helper.connect()

docs = mongo_helper.dump_image_hash()
names_in_db = [doc.file_name for doc in docs]

for n in names_in_db:
    if 'razy Ape #978' in n:
        print(f'similar {n}')

for basename in files:
    # basename = basename.encode('UTF-8').decode('UTF-8')
    if basename in names_in_db:
        print(f'{basename} exist')
    else:
        print(f'cannot find {basename}')

with open('test_filename.txt', 'w', encoding='UTF-8') as f:
    f.writelines('\n'.join(files))