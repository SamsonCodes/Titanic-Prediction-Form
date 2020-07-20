from flask import Flask, render_template, request
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from werkzeug.datastructures import ImmutableMultiDict

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('form.html', prediction_text="To survive or not to survive, that is the question.", input={})


@app.route('/predict',methods=['POST'])
def predict():
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model('xgb_titanic.model')
    input_df = get_input_df(request.form)
    prepared_input = prepare(input_df)
    prediction = loaded_model.predict(prepared_input)[0]
    message = get_prediction_text(prediction, request.form['Name'])

    return render_template('form.html', prediction_text=message, input=request.form.to_dict())


def get_input_df(form):
    input_dict = {}
    input_dict['Sex'] = form['Sex']
    input_dict['Pclass'] = int(form['Pclass'])
    input_dict['Name'] = form['Name']
    input_dict['SibSp'] = int(form['SibSp'])
    input_dict['Cabin'] = form['Cabin']
    new_input_df = pd.DataFrame(columns=['Sex', 'Pclass', 'Name', 'SibSp', 'Cabin'])
    new_input_df.loc[1] = pd.Series(input_dict)

    return new_input_df


def is_number(value):
    try:
        int(value)
        return True
    except:
        return False


def get_prediction_text(prediction, name):
    if prediction == 0:
        return "NEGATIVE: You are going to die, {}... Better stay away from that boat!".format(name)
    else:
        return "POSITIVE: You are going to survive, {}. But you might still want to stay away from that boat.".format(name)


def prepare(df):
    new_df = df.copy()
    new_df = convert_name(new_df)
    new_df = add_deck(new_df)
    new_df = fill_null(new_df)
    new_df = encode_labels(new_df)

    return new_df


def convert_name(df):
    new_df = df.copy()
    stripped_names = []
    for name in new_df["Name"]:
        segments = name.split()
        honorifics = []
        for segment in segments:
            if segment.endswith('.'):
                honorifics.append(segment)
        if len(honorifics) > 0:
            stripped_names.append(honorifics[0])
        else:
            stripped_names.append("Mrs.")
    new_df["Name"] = stripped_names

    return new_df


def add_deck(df):
    new_df = df.copy()
    new_df["Deck"] = [cabin[0] if isinstance(cabin, str) else math.nan for cabin in new_df["Cabin"]]
    new_df.drop("Cabin", axis=1, inplace=True)

    return new_df

def fill_null(df):
    new_df = df.copy()
    for column in new_df.columns:
        if new_df[column].dtype.name == "object":
            new_df[column].fillna("MISSING", inplace=True)
        else:
            new_df[column].fillna(-1, inplace=True)

    return new_df


def encode_labels(df):
    new_df = df.copy()
    loaded_classes = load_obj('label_classes')
    for column in new_df.columns:
        if new_df[column].dtype=='object':
            label_encoder = LabelEncoder()
            label_encoder.fit(list(loaded_classes[column]))
            new_df[column] = label_encoder.transform(list(new_df[column].values))

    return new_df


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
