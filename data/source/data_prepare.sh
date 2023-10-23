python get_dataset.py
python deduplicate.py --data_in ./source_data/tulu_dolly.json --data_out ./source_data/tulu_dolly.json
python deduplicate.py --data_in ./source_data/cnn_dailymail.json --data_out ./source_data/cnn_dailymail.json
python deduplicate.py --data_in ./source_data/tulu_oasst1.json --data_out ./source_data/tulu_oasst1.json
python deduplicate.py --data_in ./source_data/tulu_open_orca.json --data_out ./source_data/tulu_open_orca.json
python deduplicate.py --data_in ./source_data/tulu_code_alpaca.json --data_out ./source_data/tulu_code_alpaca.json
python deduplicate.py --data_in ./source_data/tulu_gpt4_alpaca.json --data_out ./source_data/tulu_gpt4_alpaca.json
python deduplicate.py --data_in ./source_data/tulu_sharegpt.json --data_out ./source_data/tulu_sharegpt.json
python merge.py