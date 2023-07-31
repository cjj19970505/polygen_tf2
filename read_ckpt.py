# Usage is from https://github.com/tensorflow/tensorflow/blob/a3e2c692c18649329c4210cf8df2487d2028e267/tensorflow/python/tools/inspect_checkpoint.py#L72C5-L72C65

import tensorflow as tf
import modules



def restore_attention():
    pass


if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)

    reader = tf.train.load_checkpoint("data/pretrained/vertex_model/")
    var_to_shape_map = reader.get_variable_to_shape_map()
    vertex_module_config = dict(
        decoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=24,
            dropout_rate=0.4,
            # re_zero=True,
            # memory_efficient=True,
        ),
        quantization_bits=8,
        class_conditional=True,
        max_num_input_verts=5000,
        use_discrete_embeddings=True,
    )

    vertex_model = modules.VertexModel(**vertex_module_config)
    print("asdf")
