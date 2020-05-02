import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from keras import backend as K

from keras.models import *
from keras.layers import *

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
             ]

def freeze_session(session, keep_var_names=None, output_name=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_name or []
        output_names += [v.op.name for v in  tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph


def get_ocr_gru_model(model_path):
    width, height, n_len, n_class = 164, 48, 7, len(chars)+ 1
    rnn_size = 256
    input_tensor = Input((164, 48, 3))
    x = input_tensor
    base_conv = 32
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model

def convert_gru_model_to_pb():
    wkdir = './'
    pb_filename = "ocr_plate_all_gru.pb"
    K.set_learning_phase(0)
    model = get_ocr_gru_model("./ocr_plate_all_gru.h5")
    frozen_graph = freeze_session(K.get_session(), output_name=[out.op.name for out in model.outputs])
    tf.io.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)
    print(model.inputs)
    print(model.outputs)


if __name__ == '__main__':
    convert_gru_model_to_pb()
