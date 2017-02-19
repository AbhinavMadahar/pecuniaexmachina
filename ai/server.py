from flask import Flask, request
from main import recommendation

app = Flask(__name__)

@app.route('/api/recommendation', methods=['GET'])
def show_post():
    tick = request.args.get('tick')
    highs = request.args.get('highs')
    print [float(high) for high in highs.split(",")]
    decision = recommendation(tick, [float(high) for high in highs.split(",")])
    return str(decision)

if __name__ == '__main__':
    app.run(debug=True)
