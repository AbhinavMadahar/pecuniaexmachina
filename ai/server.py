from flask import Flask, request, render_template, send_from_directory, current_app
from main import recommendation

app = Flask(__name__, static_url_path='/')

@app.route('/api/recommendation', methods=['GET'])
def show_post():
    tick = request.args.get('tick')
    highs = request.args.get('highs')
    print [float(high) for high in highs.split(",")]
    decision = recommendation(tick, [float(high) for high in highs.split(",")])
    return str(decision)

@app.route('/')
def index():
    return current_app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
