from datasets import concatenate_datasets

def merge_datasets(dataset_list):

    merged = concatenate_datasets(dataset_list)

    print("Merged dataset size:", len(merged))

    return merged