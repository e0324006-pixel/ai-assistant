from chatbot.inference import load_chatbot

chatbot = load_chatbot("./models/trained_model")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    response = chatbot(user, max_new_tokens=100)

    print("AI:", response[0]["generated_text"])