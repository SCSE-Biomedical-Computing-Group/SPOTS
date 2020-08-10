from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Conv2D, Lambda, LSTM
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from attention_tanh import AttentionLayerTanh
from attention_base import AttentionLayerBase
from tensorflow.python.keras import backend as K
# from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.keras import optimizers

def custom_loss(y_true, y_pred):
    return(K.categorical_crossentropy(y_true, y_pred, from_logits=True))

def define_attn_model_w_cnn(num_filters, hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize, attn_layer_type=0):
    """ Defining a NMT model """

    # Define an input sequence and process it.
    if batch_size:
        cnn_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize, 1), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps, fr_vsize), name='decoder_inputs')
    else:
        cnn_inputs = Input(shape=(1, en_timesteps, en_vsize), name='encoder_inputs')
        decoder_inputs = Input(shape=(fr_timesteps, fr_vsize), name='decoder_inputs')

    #CNN filters
    cnn_layer = Conv2D(filters = num_filters, kernel_size=(1, en_vsize), strides=(1, 1), padding='valid', data_format="channels_last")
    cnn_out = cnn_layer(cnn_inputs)
    sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
    encoder_inputs = sq_layer(cnn_out)
    
    # Encoder LSTM
    encoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_out, enc_state_h, enc_state_c = encoder_lstm(encoder_inputs)

    encoder_state = [enc_state_h, enc_state_c]

    #print ('K.shape', K.shape(enc_state_h), K.shape(enc_state_c))
    # Set up the decoder LSTM, using `encoder_states` as initial state.
    decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_out, dec_state_h, dec_state_c  = decoder_lstm(decoder_inputs, initial_state=encoder_state)

    # Attention layer
    if attn_layer_type == 0:
        attn_layer = AttentionLayerBase(name='attention_layer')
    elif attn_layer_type == 1:
        attn_layer = AttentionLayerTanh(name='attention_layer')
    
    attn_out, attn_states = attn_layer([encoder_out, decoder_out])

    # Concat attention input and decoder GRU output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

    # Dense layer
    dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense_time(decoder_concat_input)

    # Full model
    full_model = Model(inputs=[cnn_inputs, decoder_inputs], outputs=decoder_pred)
    opt = optimizers.Adam(clipnorm=1.)#(lr=LEARNING_RATE, clipnorm=1., decay=LEARNING_RATE/EPOCH)
    full_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    full_model.summary(line_length=200)

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    cnn_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize, 1), name='cnn_inf_inputs')
    cnn_inf_out = cnn_layer(cnn_inf_inputs)
    encoder_inf_inputs = sq_layer(cnn_inf_out)
    encoder_inf_out, enc_inf_state_h, enc_inf_state_c = encoder_lstm(encoder_inf_inputs)
    encoder_model = Model(inputs=cnn_inf_inputs, outputs=[encoder_inf_out, enc_inf_state_h, enc_inf_state_c])

    """ Decoder (Inference) model """
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
    encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, hidden_size), name='encoder_inf_states')
    decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')
    decoder_init_state_h = Input(batch_shape=(batch_size, hidden_size), name='decoder_init_h')
    decoder_init_state_c = Input(batch_shape=(batch_size, hidden_size), name='decoder_init_c')

    decoder_inf_out, decoder_inf_state_h, decoder_inf_state_c = decoder_lstm(decoder_inf_inputs, initial_state=[decoder_init_state_h, decoder_init_state_c])
    attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
    decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state_h, decoder_init_state_c, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state_h, decoder_inf_state_c])

    return full_model, encoder_model, decoder_model, attn_layer, attn_states


if __name__ == '__main__':

    """ Checking nmt model for toy examples """
    define_nmt(64, None, 20, 30, 20, 20)
