from typing import Any
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import data_utils
import os


class TransformerDecoder(tf.Module):
    def __init__(
        self,
        inputs_dim,
        context_dim=None,  # None if no sequantial context
        hidden_size=256,
        fc_size=1024,
        num_heads=4,
        layer_norm=True,
        num_layers=8,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=False,
        name="transformer_decoder",
    ):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.fc_size = fc_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.re_zero = re_zero
        self.memory_efficient = memory_efficient

        inputs = tf.keras.Input([None, inputs_dim], dtype=tf.float32)
        if context_dim is not None:
            sequential_context_embeddings = tf.keras.Input(
                [None, context_dim], dtype=tf.float32
            )
        else:
            sequential_context_embeddings = None

        x = inputs
        for layer_num in range(self.num_layers):
            # Multihead self-attention
            res = x
            if self.memory_efficient:
                # TODO
                pass
            else:
                if self.layer_norm:
                    res = tf.keras.layers.LayerNormalization()(res)
                res = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_size // self.num_heads,
                    value_dim=self.hidden_size // self.num_heads,
                    output_shape=tf.TensorShape([self.hidden_size]),
                )(query=res, value=res, use_causal_mask=True)
                if self.re_zero:
                    # TODO
                    pass
                if dropout_rate:
                    res = tf.keras.layers.Dropout(rate=dropout_rate)(res)
                x += res

            # Optional cross attention into sequential context
            if sequential_context_embeddings is not None:
                res = x
                if self.layer_norm:
                    res = tf.keras.layers.LayerNormalization()(res)
                res = tf.keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_size // self.num_heads,
                    value_dim=self.hidden_size // self.num_heads,
                    output_shape=tf.TensorShape([self.hidden_size]),
                )(query=res, value=sequential_context_embeddings)
                if self.re_zero:
                    # TODO
                    pass
                if dropout_rate:
                    res = tf.keras.layers.Dropout(rate=dropout_rate)(res)
                x += res

            # FC layers
            res = x
            if self.layer_norm:
                res = tf.keras.layers.LayerNormalization()(res)
            res = tf.keras.layers.Dense(units=self.fc_size, activation="relu")(res)
            res = tf.keras.layers.Dense(units=self.hidden_size)(res)
            if self.re_zero:
                # TODO
                pass
            if dropout_rate:
                res = tf.keras.layers.Dropout(rate=dropout_rate)(res)
            x += res

        if self.layer_norm:
            output = tf.keras.layers.LayerNormalization()(x)
        else:
            output = x

        inputs_dict = {"inputs": inputs}
        if sequential_context_embeddings is not None:
            inputs_dict["sequential_context_embeddings"] = sequential_context_embeddings

        self.model = tf.keras.Model(inputs=inputs_dict, outputs=output)

    @tf.Module.with_name_scope
    def __call__(
        self, inputs, sequential_context_embeddings=None, training=False, cache=None
    ):
        inputs_dict = {"inputs": inputs}
        if sequential_context_embeddings is not None:
            inputs_dict["sequential_context_embeddings"] = sequential_context_embeddings
        return self.model(inputs_dict, training=training)


def dequantize_verts(verts, n_bits, add_noise=False):
    """Quantizes vertices and outputs integers with specified n_bits."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2**n_bits - 1
    verts = tf.cast(verts, tf.float32)
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += tf.random_uniform(tf.shape(verts)) * (1 / float(range_quantize))
    return verts


def quantize_verts(verts, n_bits):
    """Dequantizes integer vertices to floats."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2**n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return tf.cast(verts_quantize, tf.int32)


def top_k_logits(logits, k):
    """Masks logits such that logits not in top-k are small."""
    if k == 0:
        return logits
    else:
        values, _ = tf.math.top_k(logits, k=k)
        k_largest = tf.reduce_min(values)
        logits = tf.where(
            tf.less_equal(logits, k_largest), tf.ones_like(logits) * -1e9, logits
        )
        return logits


def top_p_logits(logits, p):
    """Masks logits using nucleus (top-p) sampling."""
    if p == 1:
        return logits
    else:
        logit_shape = tf.shape(logits)
        seq, dim = logit_shape[1], logit_shape[2]
        logits = tf.reshape(logits, [-1, dim])
        sort_indices = tf.argsort(logits, axis=-1, direction="DESCENDING")
        probs = tf.gather(tf.nn.softmax(logits), sort_indices, batch_dims=1)
        cumprobs = tf.cumsum(probs, axis=-1, exclusive=True)
        # The top 1 candidate always will not be masked.
        # This way ensures at least 1 indices will be selected.
        sort_mask = tf.cast(tf.greater(cumprobs, p), logits.dtype)
        batch_indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(logits)[0]), axis=-1), [1, dim]
        )
        top_p_mask = tf.scatter_nd(
            tf.stack([batch_indices, sort_indices], axis=-1),
            sort_mask,
            tf.shape(logits),
        )
        logits -= top_p_mask * 1e9
        return tf.reshape(logits, [-1, seq, dim])


class VertexModel(tf.Module):
    def __init__(
        self,
        decoder_config,
        quantization_bits,
        class_conditional=False,
        num_classes=55,
        max_num_input_verts=2500,
        use_discrete_embeddings=True,
        name="vertex_model",
    ):
        super().__init__(name=name)
        self.embedding_dim = decoder_config["hidden_size"]
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_num_input_verts = max_num_input_verts
        self.quantization_bits = quantization_bits
        self.use_discrete_embeddings = use_discrete_embeddings

        self.decoder = TransformerDecoder(inputs_dim=self.embedding_dim ,**decoder_config)

        # Prepare context:
        if self.class_conditional:
            self.global_context_embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.num_classes,
                output_dim=self.embedding_dim,
                embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                name="class_label_embedding",
            )
        else:
            self.global_context_embedding_layer = None

        self.coord_embedding = tf.keras.layers.Embedding(
            input_dim=3,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="coord_embedding",
        )
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=self.max_num_input_verts,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name="pos_embedding",
        )

        if self.use_discrete_embeddings:
            self.vert_embedding = tf.keras.layers.Embedding(
                input_dim=2**self.quantization_bits + 1,
                output_dim=self.embedding_dim,
                embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                name="value_embeddings",
            )
        else:
            self.vert_embedding = tf.keras.layers.Dense(
                self.embedding_dim, name="value_embeddings"
            )

        if self.global_context_embedding_layer is None:
            self.zero_embed = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=[1, 1, self.embedding_dim]),
                name="embed_zero",
            )

        self.project_to_logits = tf.keras.layers.Dense(
            2**self.quantization_bits + 1,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name="project_to_logits",
        )

    def _prepare_context(self, context):
        if self.class_conditional:
            global_context_embedding = self.global_context_embedding_layer(
                context["class_label"]
            )
        else:
            global_context_embedding = None
        return global_context_embedding, None

    def _embed_inputs(self, vertices, global_context_embedding=None):
        # Dequantize inputs and get shapes
        input_shape = tf.shape(vertices)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Coord indicators (x, y, z)
        coord_embeddings = self.coord_embedding(tf.math.mod(tf.range(seq_length), 3))

        # Position embeddings
        pos_embeddings = self.pos_embedding(tf.math.floordiv(tf.range(seq_length), 3))

        # Discrete vertex value embeddings
        if self.use_discrete_embeddings:
            vert_embeddings = self.vert_embedding(vertices)
        # Continuous vertex value embeddings
        else:
            vert_embeddings = dequantize_verts(
                vertices[..., None], self.quantization_bits
            )
            vert_embeddings = self.vert_embedding(vert_embeddings)

        # TODO 这里没看懂
        # Step zero embeddings
        if global_context_embedding is None:
            zero_embed_tiled = tf.tile(self.zero_embed, [batch_size, 1, 1])
        else:
            zero_embed_tiled = global_context_embedding[:, None]

        # Aggregate embeddings
        embeddings = vert_embeddings + (coord_embeddings + pos_embeddings)[None]
        embeddings = tf.concat([zero_embed_tiled, embeddings], axis=1)

        return embeddings

    def _create_dist(
        self,
        vertices,
        global_context_embedding=None,
        sequential_context_embeddings=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        training=False,
    ):
        decoder_inputs = self._embed_inputs(vertices, global_context_embedding)

        outputs = self.decoder(
            decoder_inputs,
            sequential_context_embeddings=sequential_context_embeddings,
            training=training,
            cache=None,
        )

        logits = self.project_to_logits(outputs)
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        cat_dist = tfp.distributions.Categorical(logits=logits)
        return cat_dist
    
    @tf.Module.with_name_scope
    def __call__(self, batch, training=False):
        global_context, seq_context = self._prepare_context(batch)
        pred_dist = self._create_dist(
            batch["vertices_flat"][:, :-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context,
            training=training,
        )
        return pred_dist
    
    @tf.Module.with_name_scope
    def sample(
        self,
        num_samples,
        context=None,
        max_sample_length=None,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        recenter_verts=True,
        only_return_complete=True,
    ):
        global_context, seq_context = self._prepare_context(context)

        if global_context is not None:
            num_samples = tf.minimum(num_samples, tf.shape(global_context)[0])
            global_context = global_context[:num_samples]

        samples = tf.zeros([num_samples, 0], dtype=tf.int32)
        max_sample_length = max_sample_length or self.max_num_input_verts

        for i in tf.range(max_sample_length * 3 + 1):
            if not tf.reduce_any(tf.reduce_all(tf.not_equal(samples, 0), axis=-1)):
                break

            cat_dist = self._create_dist(
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            next_sample = cat_dist.sample()
            next_sample = next_sample[:, -1:]
            samples = tf.concat([samples, next_sample], axis=1)

        v = samples
        # Check if samples completed. Samples are complete if the stopping token
        # is produced.
        completed = tf.reduce_any(tf.equal(v, 0), axis=-1)

        # Get the number of vertices in the sample. This requires finding the
        # index of the stopping token. For complete samples use to argmax to get
        # first nonzero index.
        stop_index_completed = tf.argmax(
            tf.cast(tf.equal(v, 0), tf.int32), axis=-1, output_type=tf.int32
        )

        # For incomplete samples the stopping index is just the maximum index.
        stop_index_incomplete = (
            max_sample_length * 3 * tf.ones_like(stop_index_completed)
        )
        stop_index = tf.where(completed, stop_index_completed, stop_index_incomplete)
        num_vertices = tf.math.floordiv(stop_index, 3)

        # Convert to 3D vertices by reshaping and re-ordering x -> y -> z
        v = v[:, : (tf.reduce_max(num_vertices) * 3)] - 1
        verts_dequantized = dequantize_verts(v, self.quantization_bits)
        vertices = tf.reshape(verts_dequantized, [num_samples, -1, 3])
        vertices = tf.stack(
            [vertices[..., 2], vertices[..., 1], vertices[..., 0]], axis=-1
        )

        # Pad samples to max sample length. This is required in order to concatenate
        # Samples across different replicator instances. Pad with stopping tokens
        # for incomplete samples.
        pad_size = max_sample_length - tf.shape(vertices)[1]
        vertices = tf.pad(vertices, [[0, 0], [0, pad_size], [0, 0]])

        # 3D Vertex mask
        vertices_mask = tf.cast(
            tf.range(max_sample_length)[None] < num_vertices[:, None], tf.float32
        )

        if recenter_verts:
            vert_max = tf.reduce_max(
                vertices - 1e10 * (1.0 - vertices_mask)[..., None], keepdims=True
            )
            vert_min = tf.reduce_min(
                vertices + 1e10 * (1.0 - vertices_mask)[..., None], keepdims=True
            )
            vert_centers = 0.5 * (vert_max + vert_min)
            vertices -= vert_centers
        vertices *= vertices_mask[..., None]

        if only_return_complete:
            vertices = tf.boolean_mask(vertices, completed)
            num_vertices = tf.boolean_mask(num_vertices, completed)
            vertices_mask = tf.boolean_mask(vertices_mask, completed)
            completed = tf.boolean_mask(completed, completed)

        # Outputs
        outputs = {
            "completed": completed,
            "vertices": vertices,
            "num_vertices": num_vertices,
            "vertices_mask": vertices_mask,
        }
        return outputs
    
class FaceModel(tf.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 class_conditional=True,
                 num_classes=55,
                 decoder_cross_attention=True,
                 use_discrete_vertex_embeddings=True,
                 quantization_bits=8,
                 max_seq_length=5000,
                 name='face_model'):
        super().__init__(name=name)
        self.embedding_dim = decoder_config['hidden_size']
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.decoder_cross_attention = decoder_cross_attention
        self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings
        self.quantization_bits = quantization_bits
        

        self.class_label_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.num_classes,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.GlorotUniform(),
            name='class_label'
        )

        # vertices embedding
        if self.use_discrete_vertex_embeddings:
            self.vertices_embedding_layers = []
            for c in range(3):
                self.vertices_embedding_layers.append(
                    tf.keras.layers.Embedding(
                    input_dim=256,
                    output_dim=self.embedding_dim,
                    embeddings_initializer=tf.keras.initializers.GlorotUniform(),
                    name='coord_{}'.format(c)
                    )
                )
        else:
            self.vertices_embedding_layer = tf.keras.layers.Dense(
                units=self.embedding_dim,
                name='vertex_embeddings'
            )

        self.stopping_embeddings = tf.Variable(
            initial_value=tf.keras.initializers.GlorotUniform()(shape=[1, 2, self.embedding_dim]),
            name="stopping_embeddings"
        )

        # inputs_dim here is the dim of vertices embeddings
        self.decoder = TransformerDecoder(inputs_dim=self.embedding_dim, **decoder_config)
        self.encoder = TransformerDecoder(inputs_dim=self.embedding_dim, **encoder_config)

        # Input embeddings

        self.pos_embeddings_layer = tf.keras.layers.Embedding(
            input_dim = self.max_seq_length,
            output_dim=self.embedding_dim,
            embeddings_initializer = tf.keras.initializers.GlorotUniform,
            name='coord_embeddings'
        )

        self.project_to_pointers_layer = tf.keras.layers.Dense(
            units=self.embedding_dim,
            kernel_initializer=tf.keras.initializers.Zeros(),
            name='project_to_pointers'
        )

        if not self.class_conditional:
            self.zero_embed = tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=[1, 1, self.embedding_dim]),
                name="embed_zero"
            )
            

    def _embed_vertices(self, vertices, vertices_mask, training=False):
        if self.use_discrete_vertex_embeddings:
            vertex_embeddings = 0.
            verts_quantized = quantize_verts(vertices, self.quantization_bits)
            for c, vertices_embedding_layer in enumerate(self.vertices_embedding_layers):
                vertex_embeddings += vertices_embedding_layer(verts_quantized[...,c])
        else:
            vertex_embeddings = self.vertices_embedding_layer(vertices)
        vertex_embeddings *= vertices_mask[..., None]
        stopping_embeddings = tf.tile(self.stopping_embeddings, [tf.shape(vertices)[0], 1, 1])
        vertex_embeddings = tf.concat([stopping_embeddings, vertex_embeddings], axis=1)
        vertex_embeddings = self.encoder(vertex_embeddings, training=training)
        return vertex_embeddings


    def _prepare_context(self, context, training=False):
        if self.class_conditional:
            global_context_embedding = self.class_label_embedding_layer(context['class_label'])
        else:
            global_context_embedding = None

        vertex_embeddings = self._embed_vertices(
            context['vertices'],
            context['vertices_mask'],
            training=training
        )

        if self.decoder_cross_attention:
            sequential_context_embeddings = (vertex_embeddings * tf.pad(context['vertices_mask'], [[0, 0], [2, 0]], constant_values=1)[..., None])
        else:
            sequential_context_embeddings = None

        return (vertex_embeddings, global_context_embedding, sequential_context_embeddings)
    
    def _embed_inputs(self, faces_long, vertex_embeddings, global_context_embedding=None):
        # Face value embeddings are gathered vertex embeddings
        face_embeddings = tf.gather(vertex_embeddings, faces_long, batch_dims=1)

        pos_embeddings = self.pos_embeddings_layer(tf.range(tf.shape(faces_long)[1]))
        
        # Step zero embeddings
        batch_size = tf.shape(face_embeddings)[0]
        if global_context_embedding is None:
            zero_embed_tiled = tf.tile(self.zero_embed, [batch_size, 1, 1])
        else:
            zero_embed_tiled = global_context_embedding[:, None]

        embeddings = face_embeddings + pos_embeddings[None]
        embeddings = tf.concat([zero_embed_tiled, embeddings], axis=1)

        return embeddings

    def _create_dist(self,
                     vertex_embeddings,
                     vertices_mask,
                     faces_long,
                     global_context_embedding=None,
                     sequential_context_embeddings=None,
                     temperature=1.,
                     top_k=0,
                     top_p=1.,
                     training=False):
        decoder_inputs = self._embed_inputs(
            faces_long, vertex_embeddings, global_context_embedding)
        
        # Pass through Transformer decoder
        decoder_outputs = self.decoder(
            decoder_inputs,
            sequential_context_embeddings=sequential_context_embeddings,
            training=training
        )

        # Get pointers
        pred_pointers = self.project_to_pointers_layer(decoder_outputs)

        # TODO 看不懂
        # Get logits and mask
        logits = tf.matmul(pred_pointers, vertex_embeddings, transpose_b=True)
        logits /= tf.sqrt(float(self.embedding_dim))
        f_verts_mask = tf.pad(vertices_mask, [[0,0], [2,0]], constant_values=1.)[:, None]
        logits *= f_verts_mask
        logits -= (1. - f_verts_mask) * 1e9
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        return tfp.distributions.Categorical(logits=logits)
    
    @tf.Module.with_name_scope
    def __call__(self, batch, training=False):
        vertex_embeddings, global_context, seq_context = self._prepare_context(batch, training=training)
        pred_dict = self._create_dist(
            vertex_embeddings,
            batch['vertices_mask'],
            batch['faces'][:,:-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context,
            training=training
        )
        return pred_dict
    
    @tf.Module.with_name_scope
    def sample(self,
               context,
               max_sample_length=None,
               temperature=1.,
               top_k=0,
               top_p=0,
               only_return_complete=True):
        vertex_embeddings, global_context, seq_context = self._prepare_context(
            context, training=False
        )
        num_samples = tf.shape(vertex_embeddings)[0]

        samples = tf.zeros([num_samples, 0], dtype=tf.int32)
        for i in tf.range(max_sample_length):
            if not tf.reduce_any(tf.reduce_all(tf.not_equal(samples, 0), axis=-1)):
                break
            pred_dist = self._create_dist(
                vertex_embeddings,
                context['vertices_mask'],
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            next_sample = pred_dist.sample()[:, -1:]
            samples = tf.concat([samples, next_sample], axis=1)
        
        f = samples

        # Record completed samples
        complete_samples = tf.reduce_any(tf.equal(f, 0), axis=-1)
        
        # Find number of faces
        sample_length = tf.shape(f)[-1]

        # Get largest new face (1) index as stopping point for incomplete samples.
        max_one_ind = tf.reduce_max(
            tf.range(sample_length)[None] * tf.cast(tf.equal(f, 1), tf.int32),
            axis=-1
        )
        zero_inds = tf.cast(tf.argmax(tf.cast(tf.equal(f, 0), tf.int32), axis=-1), tf.int32)
        num_face_indices = tf.where(complete_samples, zero_inds, max_one_ind) + 1

        # mask faces beyond stopping token with zeros
        # This mask has a -1 in order to replace the last new face token with zero
        faces_mask = tf.cast(tf.range(sample_length)[None] < num_face_indices[:, None] - 1, tf.int32)
        f *= faces_mask
        # This is the real mask
        faces_mask = tf.cast(tf.range(sample_length)[None] < num_face_indices[:, None], tf.int32)

        # Pad to maximum size with zeros
        pad_size = max_sample_length - sample_length
        f = tf.pad(f, [[0,0], [0, pad_size]])

        if only_return_complete:
            f = tf.boolean_mask(f, complete_samples)
            num_face_indices = tf.boolean_mask(num_face_indices, complete_samples)
            context = tf.nest.map_structure(
                lambda x: tf.boolean_mask(x, complete_samples), context
            )
            complete_samples = tf.boolean_mask(complete_samples, complete_samples)
        
        # outputs
        outputs = {
            'context': context,
            'completed': complete_samples,
            'faces': f,
            'num_face_indices': num_face_indices,
        }
        return outputs
        

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

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
    data_utils.plot_meshes(mesh_list, ax_lims=0.4)

    vertex_model_dataset = data_utils.make_vertex_model_dataset(
        synthetic_dataset, apply_random_shift=False
    )
    vertex_model_dataset = vertex_model_dataset.repeat()
    vertex_model_dataset = vertex_model_dataset.padded_batch(
        4, padded_shapes=tf.compat.v1.data.get_output_shapes(vertex_model_dataset)
    )
    vertex_model_dataset = vertex_model_dataset.prefetch(1)

    vertex_model = VertexModel(
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
    vertex_model_batch = next(iter(vertex_model_dataset))
    vertex_samples = vertex_model.sample(
        4,
        context=vertex_model_batch,
        max_sample_length=200,
        top_p=0.95,
        recenter_verts=False,
        only_return_complete=False,
    )
