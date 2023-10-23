# Nips_challenge
Project for nips challenge submission

# How to Reproduce it

1. Get data

Get dataset through this:
```
cd ./data && python get_dataset.py && cd ..
```

Or you can go along the procedure of getting the dataset through:  
```
cd ./data/source && sh data_prepare.sh && cd ..
```
note: It's not recommanded to do so bacuase it will take a few hours.


2. Train

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd ./LLaMA-Factory/ && git checkout 7de7174 && cd ..
sh train.sh
```