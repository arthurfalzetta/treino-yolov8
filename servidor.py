from flask import Flask, request, jsonify
from app import handler  # sua função Lambda

app = Flask(__name__)

@app.route("/run", methods=["POST"])
def run():
    event = request.json or {}
    result = handler(event, None)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
