from datasets import load_dataset

def load_all_datasets():

    datasets = []

    # Dataset 1
    luth = load_dataset("kurakurai/luth-sft")

    luth_merged = (
        luth["luth_scholar"]
        .select(range(len(luth["luth_scholar"])))
    )

    datasets.append(luth_merged)

    # Dataset 2
    hardgen = load_dataset("Bingguang/HardGen", split="train")
    datasets.append(hardgen)

    # Dataset 3
    skywork = load_dataset(
        "Skywork/Skywork-Reward-Preference-80K-v0.2",
        split="train"
    )
    datasets.append(skywork)

    # Dataset 4
    autoif = load_dataset(
        "Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs",
        split="train"
    )
    datasets.append(autoif)

    return datasets