from transformers import TFT5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import tensorflow as tf

def split_gpt2_model(model, split_encoder_end, split_decoder_start):
    client_top = tf.keras.Sequential(list(model.encoder.block[:split_encoder_end]))
    client_bottom = tf.keras.Sequential(list(model.decoder.block[split_decoder_start:]), model.lm_head) #
    return client_top, client_bottom

split_start=[3]
split_end=[10]

full_model = TFT5ForConditionalGeneration.from_pretrained('./google-t5/t5-base')
client_models_top, client_models_bottom = split_gpt2_model(full_model, split_start[0], split_end[0])

print("full_model")
print(full_model)
print("client_models_top")
print(client_models_top)
print("client_models_bottom")
print(client_models_bottom)