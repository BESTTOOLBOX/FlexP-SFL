from datasets import concatenate_datasets, load_dataset
import json

# Define MMLU tasks
mmlu_tasks = ['professional_psychology', 'high_school_psychology', 'professional_law', 'moral_scenarios', 'miscellaneous']

# Load datasets and merge
data_path="/new_disk/houyz/gjx_SplitFM/mmlu_dataset/"
client_datasets = []
for task in mmlu_tasks:
    #dataset = load_dataset("cais/mmlu", task)
    test_dataset=load_dataset("parquet", data_files=data_path+task+'/'+'test-00000-of-00001.parquet')
    #print(test_dataset)
    #validation_dataset = dataset['validation']
    validation_dataset=load_dataset("parquet", data_files=data_path+task+'/'+'validation-00000-of-00001.parquet')
    #print(validation_dataset)
    #dev_dataset = dataset['dev']
    dev_dataset=load_dataset("parquet", data_files=data_path+task+'/'+'dev-00000-of-00001.parquet')
    #print(dev_dataset)
    # Concatenate test, validation, and dev datasets
    merged_dataset =concatenate_datasets([test_dataset['train'], validation_dataset['train'], dev_dataset['train']])
    
    # Split dataset into train and test (70% train, 30% test)
    split_dataset = merged_dataset.train_test_split(test_size=0.3)
    client_datasets.append({
        "train": split_dataset["train"],
        "test": split_dataset["test"]
    })

# Save client_datasets into a JSON file for record-keeping
output_data = [
    {
        "train": dataset["train"].to_dict(),
        "test": dataset["test"].to_dict()
    } for dataset in client_datasets
]

# Write to JSON file
with open("client_datasets.json", "w") as file:
    json.dump(output_data, file)
