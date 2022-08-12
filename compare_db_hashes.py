import mongo
from collections import defaultdict
from tqdm import tqdm
from util import select_func, clean_filename, compute_hash_distance

if __name__ == '__main__':
    output = 'D:\\github\\detect_copymint\\temp_0730_attack\\attack_result.csv'
    attack_name = 'crop0.1'
    mongo_address = "mongodb://localhost:27017/"
    mongo_db = 'imagehash'
    mongo_collection = 'hash_crop_resistant_0810_large_nn_attack'
    mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()
    docs = mongo_helper.dump_image_hash()

    mongo_collection = 'hash_crop_resistant_0810_large_nn'
    mongo_helper = mongo.MongoHelper(mongo_address=mongo_address, mongo_db=mongo_db, mongo_collection=mongo_collection)
    mongo_helper.connect()

    docs2 = mongo_helper.dump_image_hash()

    if False:
        types = ['base', 'flip0', 'flip1', 'flip2', 'rotate0', 'rotate1', 'rotate2']
    else:
        types = ['base']


    all_dist_result = defaultdict(list)
    print('\nfind db hash')
    func_name = 'crop_resistant_hash'
    func = select_func(func_name)
    for query in tqdm(docs):
        min_n = float('inf')
        min_n_dist = float('inf')

        dist_result = []

        query_hash = query.get_hash(func_name, 'base')

        for field in docs2:
            field_hash = field.get_hash(func_name, 'base')

            # if field.file_name[:-4] == query.file_name[:-4]:
            #     print(f'{query.file_name[:-4]} start hash compare')

            dist = compute_hash_distance(query_hash.hash, field_hash.hash, func=func)
            dist_result.append((field.file_name, dist))

            # if field.file_name[:-4] == query.file_name[:-4]:
            #     print(f'{query.file_name}: dist {dist}')

        dist_result = sorted(dist_result, key=lambda x: x[1])
        # with open(os.path.join(os.path.dirname(output), 'dist_goon6776.csv'), 'w', encoding='UTF-8') as f:
        #     for dist in dist_result:
        #         f.write(f'{dist[1]}, {dist[0]}\n')
        find_list = [(i, v[1]) for i, v in enumerate(dist_result) if v[0][:-4] == query.file_name[:-4]]
        if len(find_list) == 0:
            break

        n, dist = find_list[0]
        if 0 <= n < min_n:
            min_n = n
            min_n_dist = dist
            # print(dist_result)
        all_dist_result[query.file_name].append((attack_name, min_n, min_n_dist))

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