from flask import Flask, render_template, request, redirect, url_for
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/result/<opennessScore>,<conscientiousnessScore>,<extraversionScore>,<agreeablenessScore>,<neuroticismScore>')
def result(opennessScore, conscientiousnessScore, extraversionScore, agreeablenessScore, neuroticismScore):
	opennessScore = int(round(float(opennessScore[1:7]) * 100))
	conscientiousnessScore = int(round(float(conscientiousnessScore[1:7]) * 100))
	extraversionScore = int(round(float(extraversionScore[1:7]) * 100))
	agreeablenessScore = int(round(float(agreeablenessScore[1:7]) * 100))
	neuroticismScore = int(round(float(neuroticismScore[1:7]) * 100))
	
	return render_template("homepage.html", opennessScore=opennessScore, conscientiousnessScore=conscientiousnessScore, extraversionScore=extraversionScore, agreeablenessScore=agreeablenessScore, neuroticismScore=neuroticismScore)

@app.route('/status_update', methods=['POST'])
def status_update():
	MAX_NB_WORDS = 2800
	MAX_SEQUENCE_LENGTH = 1000
	response = request.form
	texts = []
	texts.append(response['status_update'].encode('utf-8'))
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	
# 	Load Openness model
 	json_file = open('openness.json', 'r')
 	loaded_model_json = json_file.read()
 	json_file.close()
 	loaded_model = model_from_json(loaded_model_json)
 	loaded_model.load_weights("openness.h5")
 	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
 	predictions = loaded_model.predict(data)
 	opennessScore = [x[1] for x in predictions]
 	
# 	# Load Conscientiousness model
	json_file = open('conscientiousness.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("conscientiousness.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
	predictions = loaded_model.predict(data)
	conscientiousnessScore = [x[1] for x in predictions]

# 	# Load Extraversion model
	json_file = open('extraversion.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("extraversion.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
	predictions = loaded_model.predict(data)
	extraversionScore = [x[1] for x in predictions]
	
# 	# Load Agreeableness model
	json_file = open('agreeableness.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("agreeableness.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
	predictions = loaded_model.predict(data)
	agreeablenessScore = [x[1] for x in predictions]

# 	# Load Neuroticism model
	json_file = open('neuroticism.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("neuroticism.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
	predictions = loaded_model.predict(data)
	neuroticismScore = [x[1] for x in predictions]
	
	return redirect(url_for('result', opennessScore=opennessScore,conscientiousnessScore=conscientiousnessScore, extraversionScore=extraversionScore, agreeablenessScore=agreeablenessScore, neuroticismScore=neuroticismScore))
	
if __name__ == '__main__':
   app.run(debug = True)