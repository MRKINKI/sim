import pymongo
import json


class DownloadData:
    def __init__(self):
        self.client = pymongo.MongoClient('192.168.1.63', 27017)
        self.db = self.client['Food&Health']
        self.food_collection = self.db['meishij_tangzhou']

    @staticmethod
    def get_field_info(field, dic):
        info = dic[field] if field in dic else ''
        if isinstance(info, list):
            return ' '.join(info)
        else:
            return info

    @staticmethod
    def get_material(dic):
        materials = []
        if '用料' not in dic:
            return materials
        # print(dic['_id'])
        for k, sub_materials in dic['用料'].items():
            for m in sub_materials:
                materials.append(m['材料'])
        return materials

    def download_all(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as fout:
            for doc in self.food_collection.find():
                sample = {'id': str(doc['_id']),
                          'title': doc['title'],
                          'efficacy': self.get_field_info('功效', doc),
                          'illustration': self.get_field_info('简介说明', doc),
                          'practice': self.get_field_info('做法', doc),
                          'materials': self.get_material(doc)
                          }
                fout.write(json.dumps(sample) + '\n')

    def download_cho(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as fout:
            for doc in self.food_collection.find():
                if '简介说明' in doc:
                    sample = {'id': str(doc['_id']),
                              'text': doc['简介说明']}
                    fout.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    file = './data/tangzhou_all.json'
    dd = DownloadData()
    dd.download_all(file)
    # dd.download_cho(file)
