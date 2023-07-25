import tensorflow as tf
import data_utils
import os
import modules
from itertools import chain
from tqdm import tqdm
from datetime import datetime

tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)

# Prepare synthetic dataset
ex_list = []
for k, mesh in enumerate(["cube", "cylinder", "cone", "icosphere"]):
    mesh_dict = data_utils.load_process_mesh(
        os.path.join("meshes", "{}.obj".format(mesh))
    )
    mesh_dict["class_label"] = k
    ex_list.append(mesh_dict)

synthetic_dataset = tf.data.Dataset.from_generator(
    lambda: ex_list,
    output_types={"vertices": tf.int32, "faces": tf.int32, "class_label": tf.int32},
    output_shapes={
        "vertices": tf.TensorShape([None, 3]),
        "faces": tf.TensorShape([None]),
        "class_label": tf.TensorShape(()),
    },
)

# Plot the meshes
mesh_list = []
for ex in synthetic_dataset:
    mesh_list.append(
        {
            "vertices": data_utils.dequantize_verts(ex["vertices"].numpy()),
            "faces": data_utils.unflatten_faces(ex["faces"].numpy()),
        }
    )
meshes_plot = data_utils.plot_meshes(mesh_list, ax_lims=0.4)
with file_writer.as_default():
    tf.summary.image("mesh_list", meshes_plot[None], step=0)

vertex_model_dataset = data_utils.make_vertex_model_dataset(
    synthetic_dataset, apply_random_shift=False
)
vertex_model_dataset = vertex_model_dataset.repeat()
vertex_model_dataset = vertex_model_dataset.padded_batch(
    4, padded_shapes=tf.compat.v1.data.get_output_shapes(vertex_model_dataset)
)
vertex_model_dataset = vertex_model_dataset.prefetch(1)

vertex_model = modules.VertexModel(
    decoder_config={
        "hidden_size": 128,
        "fc_size": 512,
        "num_layers": 3,
        "dropout_rate": 0.0,
    },
    class_conditional=True,
    num_classes=4,
    max_num_input_verts=250,
    quantization_bits=8,
)

face_model_dataset = data_utils.make_face_model_dataset(
    synthetic_dataset ,apply_random_shift=False, shuffle_vertices=False
)
face_model_dataset = face_model_dataset.repeat()
face_model_dataset = face_model_dataset.padded_batch(
    4, padded_shapes=tf.compat.v1.data.get_output_shapes(face_model_dataset)
)
face_model_dataset = face_model_dataset.prefetch(1)

face_model = modules.FaceModel(
    encoder_config={
        "hidden_size": 128,
        "fc_size": 512,
        "num_layers": 3,
        "dropout_rate": 0.0,
    },
    decoder_config={
        "hidden_size": 128,
        "fc_size": 512,
        "num_layers": 3,
        "dropout_rate": 0.0,
    },
    class_conditional=False,
    max_seq_length=500,
    quantization_bits=8,
    decoder_cross_attention=True,
    use_discrete_vertex_embeddings=True,
)

# face_model_pred_dist = face_model(face_model_batch)
# face_model_loss = -tf.reduce_sum(
#     face_model_pred_dist.log_prob(face_model_batch["faces"])
#     * face_model_batch["faces_mask"]
# )
# face_samples = face_model.sample(
#     context=vertex_samples,
#     max_sample_length=500,
#     top_p=0.95,
#     only_return_complete=False,
# )
# print(face_model_batch)
# print(face_model_pred_dist)
# print(face_samples)

learning_rate = 5e-4
training_steps = 500
check_step = 50

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# optimizer.build(vertex_model.trainable_variables + face_model.trainable_variables)


@tf.function
def train_step(vertex_model_batch, face_model_batch, optimizer):
    # We need persistent=True to compute gradient twice.
    with tf.GradientTape(persistent=True) as tape:
        vertex_model_pred_dist = vertex_model(vertex_model_batch, training=True)
        vertex_model_loss = -tf.reduce_sum(
            vertex_model_pred_dist.log_prob(vertex_model_batch["vertices_flat"])
            * vertex_model_batch["vertices_flat_mask"]
        )

        face_model_pred_dist = face_model(face_model_batch, training=True)
        face_model_loss = -tf.reduce_sum(
            face_model_pred_dist.log_prob(face_model_batch["faces"])
            * face_model_batch["faces_mask"]
        )

    vertex_model_grads = tape.gradient(
        vertex_model_loss, vertex_model.trainable_variables
    )
    face_model_grads = tape.gradient(face_model_loss, face_model.trainable_variables)

    # Persistent GradientTape should be manually deleted
    # https://www.tensorflow.org/guide/autodiff
    # https://stackoverflow.com/a/56073840/11879605
    del tape
    optimizer.apply_gradients(
        chain(
            zip(vertex_model_grads, vertex_model.trainable_variables),
            zip(face_model_grads, face_model.trainable_variables),
        )
    )

    return vertex_model_loss, face_model_loss


pbar = tqdm(total=training_steps)
for vertex_model_batch, face_model_batch, step in zip(
    vertex_model_dataset, face_model_dataset, range(training_steps)
):
    vertex_model_loss, face_model_loss = train_step(
        vertex_model_batch, face_model_batch, optimizer
    )

    with file_writer.as_default():
        tf.summary.scalar("vertex_model_loss", vertex_model_loss, step=step)
        tf.summary.scalar("face_model_loss", face_model_loss, step=step)

    if step % check_step == 0 or step + 1 == training_steps:

        vertex_samples = vertex_model.sample(
            4,
            context=vertex_model_batch,
            max_sample_length=200,
            top_p=0.95,
            recenter_verts=False,
            only_return_complete=False,
        )

        face_samples = face_model.sample(
            context=vertex_samples,
            max_sample_length=500,
            top_p=0.95,
            only_return_complete=False,
        )

        mesh_list = []
        for n in range(4):
            mesh_list.append(
                {
                    'vertices': vertex_samples['vertices'][n][:vertex_samples['num_vertices'][n].numpy()].numpy(),
                    'faces': data_utils.unflatten_faces(
                        face_samples['faces'][n][:face_samples['num_face_indices'][n].numpy()].numpy()
                    )
                }
            )
        meshes_plot = data_utils.plot_meshes(mesh_list, ax_lims=0.5)
        with file_writer.as_default():
            tf.summary.image("mesh_plots", meshes_plot[None], step=step)
    
    pbar.update()
        
pbar.close()
