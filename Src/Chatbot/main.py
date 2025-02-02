from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random
import json


# Load the trained complex model
model = load_model('rookie.h5')

# Load the intent data
data_path = 'stride.json'  # Update the file path as necessary
with open(data_path, 'r') as file:
    data = json.load(file)
intents = data['intents']

# Initialize the tokenizer and label encoder
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text for intent in intents for text in intent['text']])
label_encoder = LabelEncoder()
labels = [intent['intent'] for intent in intents]
label_encoder.fit(labels)

# Preprocess input function
def preprocess_input(user_input):
    sequence = tokenizer.texts_to_sequences([user_input])
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)  # Adjust the length based on training
    return sequence

# Get response function
def get_response(user_input):
    # Preprocess input
    sequence = preprocess_input(user_input)

    # Predict intent
    prediction = model.predict(sequence)
    predicted_class = prediction.argmax(axis=1)
    intent = label_encoder.inverse_transform(predicted_class)

    # Find the corresponding intent and respond
    for i in intents:
        if i['intent'] == intent[0]:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

# Chat function
def chat():
    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Start the chatbot
chat()
