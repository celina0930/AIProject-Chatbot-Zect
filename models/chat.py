from models.train import *

SAVE_FILE_NM = "saved_model/18-1.75531.h5"
model.load_weights(SAVE_FILE_NM)
print(model.evaluate([index_inputs, index_outputs], index_targets))

def get_response(msg):
    test_index_inputs, _ = enc_processing([msg], char2idx)
    predict_tokens = model.inference(test_index_inputs)

    return ' '.join([idx2char[str(t)] for t in predict_tokens])


