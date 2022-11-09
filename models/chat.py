from models.train import model, index_inputs, index_outputs, index_targets, prepro_configs, enc_processing
from pykospacing import Spacing

SAVE_FILE_NM = "models/saved_model/04-1.22536.h5"
model.load_weights(SAVE_FILE_NM)
#print(model.evaluate([index_inputs, index_outputs], index_targets))

char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']

def get_response(msg):
    test_index_inputs, _ = enc_processing([msg], char2idx)
    predict_tokens = model.inference(test_index_inputs)
    outputs = ''.join([idx2char[str(o)] for o in predict_tokens])
    spacing = Spacing()
    answer = spacing(outputs)
    return answer
