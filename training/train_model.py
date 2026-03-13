from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from config.training_config import MODEL_NAME, OUTPUT_DIR
from utils.format_data import format_dataset


def train_model(dataset):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # format dataset
    dataset = dataset.map(format_dataset)
    dataset = dataset.select(range(1))

    # tokenize dataset
    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokens["labels"] = tokens["input_ids"]
        return tokens

    dataset = dataset.map(tokenize_function, batched=True)

    # remove unnecessary columns
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,     # show progress every step
        save_steps=500,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)