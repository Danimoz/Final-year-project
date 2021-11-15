from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        model=pickle.load(open("spam.pkl", "rb"))
        cv=pickle.load(open("vectorizer.pkl","rb"))
        msg = request.form.get('mlmessage')
        data=[msg]
        vect=cv.transform(data).toarray()
        prediction=model.predict(vect)
        result = prediction[0]
            
        print(result)

    return render_template('index.html', result = result)


if __name__ == "__main__":
    app.run(debug=True)