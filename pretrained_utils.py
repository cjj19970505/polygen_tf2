# Usage is from https://github.com/tensorflow/tensorflow/blob/a3e2c692c18649329c4210cf8df2487d2028e267/tensorflow/python/tools/inspect_checkpoint.py#L72C5-L72C65

import tensorflow as tf
import modules
import re


def _implement_mapping(layer_mappings, source_reader):
    for layer in layer_mappings:
        if type(layer) == tf.keras.layers.Embedding:
            layer.embeddings.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "embeddings")))
        elif type(layer) == modules.MemoryEffecientMultiheadSelfAttention:
            layer.wqkv.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "wqkv")))
            layer.wo.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "wo")))
        elif type(layer) == tf.keras.layers.MultiHeadAttention:
            layer._key_dense.kernel.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "k/kernel")).reshape(layer._key_dense.kernel.shape))
            layer._query_dense.kernel.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "q/kernel")).reshape(layer._query_dense.kernel.shape))
            layer._value_dense.kernel.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "v/kernel")).reshape(layer._value_dense.kernel.shape))
            layer._output_dense.kernel.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "output_transform/kernel")).reshape(layer._output_dense.kernel.shape))
        elif type(layer) == tf.keras.layers.Dense:
            layer.kernel.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "kernel")))
            layer.bias.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "bias")))
        elif type(layer) == tf.keras.layers.LayerNormalization:
            layer.gamma.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "layer_norm_scale")))
            layer.beta.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "layer_norm_bias")))
        elif type(layer) == modules.ReZeroLayer:
            layer.alpha.assign(source_reader.get_tensor("{}{}".format(layer_mappings[layer], "alpha")))
        else:
            layer.deref().assign(source_reader.get_tensor(layer_mappings[layer]))

def restore_vertex_model(vertex_model:modules.VertexModel, ckpt_dir:str):
    source_reader = tf.train.load_checkpoint(ckpt_dir)
    layer_mappings = {}
    layer_mappings[vertex_model.coord_embedding] = 'vertex_model/coord_embeddings/'
    layer_mappings[vertex_model.pos_embedding] = 'vertex_model/coord_embeddings_1/'
    layer_mappings[vertex_model.vert_embedding] = 'vertex_model/value_embeddings/'
    layer_mappings[vertex_model.global_context_embedding_layer] = 'vertex_model/class_label/'
    for layer_num in range(24):
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['self_attention_layers'][0]] = 'vertex_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['self_attention_layers'][1]] = 'vertex_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['self_attention_layers'][2]] = 'vertex_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['fc_layers'][0]] = 'vertex_model/transformer_decoder/layer_{}/fc/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['fc_layers'][1]] = 'vertex_model/transformer_decoder/layer_{}/fc_1/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['fc_layers'][2]] = 'vertex_model/transformer_decoder/layer_{}/fc_2/'.format(layer_num)
        layer_mappings[vertex_model.decoder.attention_layers[layer_num]['fc_layers'][3]] = 'vertex_model/transformer_decoder/layer_{}/fc/'.format(layer_num)
    layer_mappings[vertex_model.decoder.output_layer] = 'vertex_model/transformer_decoder/output/'
    layer_mappings[vertex_model.project_to_logits] = 'vertex_model/project_to_logits/'
    _implement_mapping(layer_mappings, source_reader)

def restore_face_model(face_model:modules.FaceModel, ckpt_dir:str):
    source_reader = tf.train.load_checkpoint(ckpt_dir)
    layer_mappings = {}
    layer_mappings[face_model.vertices_embedding_layers[0]] = 'face_model/coord_0/'
    layer_mappings[face_model.vertices_embedding_layers[1]] = 'face_model/coord_1/'
    layer_mappings[face_model.vertices_embedding_layers[2]] = 'face_model/coord_2/'
    layer_mappings[face_model.pos_embeddings_layer] = 'face_model/coord_embeddings/'
    layer_mappings[face_model.zero_embed.ref()] = 'face_model/embed_zero'
    layer_mappings[face_model.stopping_embeddings.ref()] = 'face_model/stopping_embeddings'
    
    for layer_num in range(14):
        layer_mappings[face_model.decoder.attention_layers[layer_num]['self_attention_layers'][0]] = 'face_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['self_attention_layers'][1]] = 'face_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['self_attention_layers'][2]] = 'face_model/transformer_decoder/layer_{}/self_attention/'.format(layer_num)

        layer_mappings[face_model.decoder.attention_layers[layer_num]['cross_attention_layers'][0]] = 'face_model/transformer_decoder/layer_{}/cross_attention/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['cross_attention_layers'][1]] = 'face_model/transformer_decoder/layer_{}/cross_attention/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['cross_attention_layers'][2]] = 'face_model/transformer_decoder/layer_{}/cross_attention/'.format(layer_num)

        layer_mappings[face_model.decoder.attention_layers[layer_num]['fc_layers'][0]] = 'face_model/transformer_decoder/layer_{}/fc/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['fc_layers'][1]] = 'face_model/transformer_decoder/layer_{}/fc_1/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['fc_layers'][2]] = 'face_model/transformer_decoder/layer_{}/fc_2/'.format(layer_num)
        layer_mappings[face_model.decoder.attention_layers[layer_num]['fc_layers'][3]] = 'face_model/transformer_decoder/layer_{}/fc/'.format(layer_num)

    layer_mappings[face_model.decoder.output_layer] = 'face_model/transformer_decoder/output/'

    for layer_num in range(10):
        layer_mappings[face_model.encoder.attention_layers[layer_num]['self_attention_layers'][0]] = 'face_model/transformer_encoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[face_model.encoder.attention_layers[layer_num]['self_attention_layers'][1]] = 'face_model/transformer_encoder/layer_{}/self_attention/'.format(layer_num)
        layer_mappings[face_model.encoder.attention_layers[layer_num]['self_attention_layers'][2]] = 'face_model/transformer_encoder/layer_{}/self_attention/'.format(layer_num)

        layer_mappings[face_model.encoder.attention_layers[layer_num]['fc_layers'][0]] = 'face_model/transformer_encoder/layer_{}/fc/'.format(layer_num)
        layer_mappings[face_model.encoder.attention_layers[layer_num]['fc_layers'][1]] = 'face_model/transformer_encoder/layer_{}/fc_1/'.format(layer_num)
        layer_mappings[face_model.encoder.attention_layers[layer_num]['fc_layers'][2]] = 'face_model/transformer_encoder/layer_{}/fc_2/'.format(layer_num)
        layer_mappings[face_model.encoder.attention_layers[layer_num]['fc_layers'][3]] = 'face_model/transformer_encoder/layer_{}/fc/'.format(layer_num)

    layer_mappings[face_model.encoder.output_layer] = 'face_model/transformer_encoder/output/'
    layer_mappings[face_model.project_to_pointers_layer] = 'face_model/project_to_pointers/'

    _implement_mapping(layer_mappings, source_reader)

    

