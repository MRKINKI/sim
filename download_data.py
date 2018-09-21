import pymongo
import json


class DownloadData:
    def __init__(self):
        self.client = pymongo.MongoClient('192.168.1.63', 27017)
        self.db = self.client['Food&Health']
        self.food_collection = self.db['meishij_tangzhou']

    def download_all(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as fout:
            for doc in self.food_collection.find():
                sample = {'id': doc['_id'],
                          'title': doc['title'],
                          'efficacy': doc['功效'],
                          'illustration': doc['简介说明']}
                fout.write(json.loads(sample) + '\n')

    def download_cho(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as fout:
            for doc in self.food_collection.find():
                if '简介说明' in doc:
                    sample = {'id': str(doc['_id']),
                              'text': doc['简介说明']}
                    fout.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    file = './data/tangzhou.json'
    dd = DownloadData()
    dd.download_cho(file)
