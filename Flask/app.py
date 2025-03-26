from flask import Flask, render_template, flash, request, url_for, redirect, session

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home(): 
    return render_template("index.html")
@app.route('/scanner', methods=['GET', 'POST'])
def about(): 
    return render_template("scanner.html")

if __name__ == "__main__":
    app.run(debug=True)
