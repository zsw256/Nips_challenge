from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    #df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return

def get_data_from_cnn_dailymail(dataset):
    dataset_formated = []
    print('Reformatting cnn_dailymail...')

    for item in tqdm(dataset['train']):
        dataset_formated.append({
            'instruction':'Summarize the below article in 3 sentences.\n',
            'input':item['article'],
            'output':item['highlights'],
            'data_source':'cnn_dailymail'
        })
    
    return dataset_formated

def get_data_from_lima(dataset):
    dataset_formated = []
    print('Reformatting lima...')

    for item in tqdm(dataset):
        data = item['conversations']
        history=[]
        for i in range(int(len(data[:-2])/2)):
            history.append([data[i],data[i+1]])
        instruction = data[-2]
        output = data[-1]
        if len(history)!=0:
            dataset_formated.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'history':history,
                'data_source':item['source']
            })
        else:
            dataset_formated.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'data_source':item['source']
            })

    return dataset_formated

def get_data_from_tulu(dataset):
    dataset_formated=[]
    print('Reformatting tulu...')

    for item in tqdm(dataset):
        data = item['messages']
        history=[]
        for i in range(int(len(data[:-2])/2)):
            history.append([data[i]['content'],data[i+1]['content']])
        instruction = data[-2]['content']
        output = data[-1]['content']

        dataset_formated.append({
            'instruction':instruction,
            'input':'',
            'output':output,
            'history':history,
            'data_source':'tulu_'+item['dataset']
        })
    return dataset_formated

def main():
    if not os.path.exists('./source_data/'):
        os.mkdir('./source_data/')
    
    dataset = load_dataset("GAIR/lima",token='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')
    dataset = get_data_from_lima(dataset['train'])
    list_dict_to_json(dataset[:-30],'./source_data/lima.json')
    list_dict_to_json(dataset[-30:],'./source_data/lima_chat.json')

    dataset = load_dataset('cnn_dailymail', '3.0.0')
    dataset = get_data_from_cnn_dailymail(dataset)[:20000]
    list_dict_to_json(dataset,'./source_data/cnn_dailymail.json')

    dataset = load_dataset("garage-bAInd/Open-Platypus")
    list_dict_to_json(dataset['train'],'./source_data/open_platypus.json')


    dataset = load_dataset("Rocinante/tulu_v1")
    tulu = get_data_from_tulu(dataset['train'])
    dataset = load_dataset("Rocinante/tulu_v2")
    tulu += get_data_from_tulu(dataset['train'])
    
    sources_needed=['tulu_sharegpt','tulu_gpt4_alpaca','tulu_oasst1','tulu_open_orca','tulu_code_alpaca','tulu_dolly']
    tulu_subsets={}
    for source in sources_needed:
        tulu_subsets[source]=[]
    print('Deviding tulu...')
    for item in tqdm(tulu):
        source = item['data_source']
        if source in sources_needed:
            tulu_subsets[source].append(item)
    print(tulu_subsets.keys())
    for source in sources_needed:
        list_dict_to_json(tulu_subsets[source],'./source_data/'+source+'.json')
    
    sources = [item['data_source'] for item in tulu]
    from collections import Counter
    result = Counter(sources)
    print(result)

main()

