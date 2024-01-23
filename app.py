from flask import Flask, request, render_template
from certibot.chat_ai import get_ai_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    return get_ai_response(user_input)

if __name__ == "__main__":
    app.run(debug=True)
