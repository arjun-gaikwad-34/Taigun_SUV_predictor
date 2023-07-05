import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request

# from ml import X_train

# de pickling the model
clf = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route("/") #decorators
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    #label encode and normalize the inputs
    features = [int(x) for x in request.form.values()]
    with open('sst.pkl', 'rb') as file:
        sst = pickle.load(file)
    # sst = sst.fit(X_train)
    output = clf.predict(sst.transform([features]))
    print(output)
    if output[0] == 0:
        pred =  "Will NOT buy SUV"

    else:
        pred =  "Will buy SUV"

    return  render_template('index.html', pred = pred)


if __name__ == "__main__":
    app.run(debug = True)