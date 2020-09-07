from flask import Flask, jsonify, request
import generator_recsys
import tensorflow as tf
import data_loader_recsys
import utils
import argparse
from tensorflow.contrib import learn
import json

app = Flask(__name__)
sess = tf.Session()

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='Data/Session/')
parser.add_argument("--dilated_channels", type=int, default=100, help='number of dilated channels')
parser.add_argument("--learning_rate", type=float, default=0.008, help='learning rate')
parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
args = parser.parse_args()
model_path = args.datapath  + "/" + "model.ckpt"
vocab_path = args.datapath + "/" + "vocab.pickle"


def load_model(n_items, path):
    model_params = {
        'item_size': n_items,
        'dilated_channels': args.dilated_channels,
        'dilations': [1, 2, 1, 2, 1, 2, ],
        'kernel_size': args.kernel_size,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'is_negsample': False
    }
    itemrec = generator_recsys.NextItNet_Decoder(model_params)
    itemrec.train_graph(model_params['is_negsample'])
    itemrec.predict_graph(model_params['is_negsample'], reuse=True)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, path)
    return itemrec

def get_dataset(path):
    vocab = learn.preprocessing.VocabularyProcessor.restore(path)
    item_dict = vocab.vocabulary_._mapping
    vocabulary = vocab.vocabulary_
    print("len item dict")
    print(len(item_dict))
    return item_dict, vocabulary, vocab




#item_dict, vocabulary,vocabprocessor  = get_dataset(vocab_path)
item_dict = json.load(open(vocab_path, 'r'))
model = load_model(len(item_dict)+1, model_path)
vocabulary = json.load(open(vocab_path+"inverted", 'r'))

def pad_sequence(user_profile, max_seq_size):

    if max_seq_size > len(user_profile):
        dif = max_seq_size - len(user_profile)
        # fill gaps with UNK (0) interaction as suggested in docs
        pads = [0] * dif
        return pads + user_profile
    # longer sequences are not a problem according to the readme
    return user_profile


def prepare_sequence(user_profile, item_dict):
    user_profile = [item_dict[str(i)] if str(i) in item_dict else -1 for i in user_profile]
    max_seq_size = 80
    user_profile = pad_sequence(user_profile, max_seq_size)
    return [user_profile]

def recommend(model, user_profile, item_dict, vocabulary, top_k=10):
    input_sequence = prepare_sequence(user_profile, item_dict)
    print("original input")
    print(user_profile)
    print("prepared input sequence")
    print(input_sequence)
    [probs] = sess.run([model.g_probs], feed_dict={model.input_predict: input_sequence})
    if probs.shape[0] > 0:
        pred_items = utils.sample_top_k_with_scores(probs[0][-1], top_k=top_k)
        predictions = [(vocabulary[str(item_token)],score) if str(item_token) in vocabulary else ("[UNK]", score) for (item_token, score) in pred_items]
        print("predictions")
        print(predictions)
        return predictions
    print("empty pred")
    return []


@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    request_data = request.json
    user_profile = request_data['user_profile']
    probs = recommend(model, user_profile, item_dict, vocabulary)
    json_serializable = [(i, str(score)) for i, score in probs]
    return jsonify({"items": json_serializable})


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
