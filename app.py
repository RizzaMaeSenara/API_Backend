from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Server is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').lower()

    if user_input == 'hi':
        return jsonify({'reply': 'Hello, Rizza Senara! How are you?'})
    elif user_input == "i'm doing good" or user_input == "im doing good":
        return jsonify({'reply': "That's great to hear! Let me know if you need anything."})
    elif user_input == "thank you":
        return jsonify({'reply': "You're welcome!"})
    else:
        return jsonify({'reply': 'Sorry, I don\'t understand'})

if __name__ == '__main__':
    app.run(debug=True)

