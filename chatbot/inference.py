from transformers import pipeline

def load_chatbot(model_path):

    chatbot = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path
    )

    return chatbot