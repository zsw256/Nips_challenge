import json


with open('../dataset/merged_data.json','r') as f:
    dataset = json.load(f)

sources = [item['data_source'] for item in dataset]

from collections import Counter
result = Counter(sources)
print(result)
