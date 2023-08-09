import tensorflow as tf
import modules
import data_utils
from pretrained_utils import restore_face_model, restore_vertex_model
from datetime import datetime

log_dir = "logs/sample_pretrained/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir)

vertex_module_config = dict(
    decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=24,
        dropout_rate=0.4,
        re_zero=True,
        memory_efficient=True,
    ),
    quantization_bits=8,
    class_conditional=True,
    max_num_input_verts=5000,
    use_discrete_embeddings=True,
)

face_module_config = dict(
    encoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=10,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
    ),
    decoder_config=dict(
        hidden_size=512,
        fc_size=2048,
        num_heads=8,
        layer_norm=True,
        num_layers=14,
        dropout_rate=0.2,
        re_zero=True,
        memory_efficient=True,
    ),
    class_conditional=False,
    decoder_cross_attention=True,
    use_discrete_vertex_embeddings=True,
    max_seq_length=8000,
)

class_id = "49) table"  # @param ['0) airplane,aeroplane,plane','1) ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin','2) bag,traveling bag,travelling bag,grip,suitcase','3) basket,handbasket','4) bathtub,bathing tub,bath,tub','5) bed','6) bench','7) birdhouse','8) bookshelf','9) bottle','10) bowl','11) bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi','12) cabinet','13) camera,photographic camera','14) can,tin,tin can','15) cap','16) car,auto,automobile,machine,motorcar','17) cellular telephone,cellular phone,cellphone,cell,mobile phone','18) chair','19) clock','20) computer keyboard,keypad','21) dishwasher,dish washer,dishwashing machine','22) display,video display','23) earphone,earpiece,headphone,phone','24) faucet,spigot','25) file,file cabinet,filing cabinet','26) guitar','27) helmet','28) jar','29) knife','30) lamp','31) laptop,laptop computer','32) loudspeaker,speaker,speaker unit,loudspeaker system,speaker system','33) mailbox,letter box','34) microphone,mike','35) microwave,microwave oven','36) motorcycle,bike','37) mug','38) piano,pianoforte,forte-piano','39) pillow','40) pistol,handgun,side arm,shooting iron','41) pot,flowerpot','42) printer,printing machine','43) remote control,remote','44) rifle','45) rocket,projectile','46) skateboard','47) sofa,couch,lounge','48) stove','49) table','50) telephone,phone,telephone set','51) tower','52) train,railroad train','53) vessel,watercraft','54) washer,automatic washer,washing machine']
num_samples_min = 1  # @param
num_samples_batch = 8  # @param
max_num_vertices = 400  # @param
max_num_face_indices = 2000  # @param
top_p_vertex_model = 0.9  # @param
top_p_face_model = 0.9  # @param

vertex_model = modules.VertexModel(**vertex_module_config)
face_model = modules.FaceModel(**face_module_config)

class_id = int(class_id.split(")")[0])
vertex_model_context = {
    "class_label": tf.fill(
        [
            num_samples_batch,
        ],
        class_id,
    )
}

# call to create model variables
dummy_vertex_samples = vertex_model.sample(
    num_samples_batch,
    context=vertex_model_context,
    max_sample_length=50,
    top_p=top_p_vertex_model,
    recenter_verts=True,
    only_return_complete=True,
)
dummy_face_samples = face_model.sample(
    dummy_vertex_samples, max_sample_length=10, top_p=0.9, only_return_complete=True
)

restore_vertex_model(vertex_model, "data/pretrained/vertex_model/")
restore_face_model(face_model, "data/pretrained/face_model/")

mesh_list = []

while len(mesh_list) < num_samples_min:
    v_samples = vertex_samples = vertex_model.sample(
        num_samples_batch,
        context=vertex_model_context,
        max_sample_length=max_num_vertices,
        top_p=top_p_vertex_model,
        recenter_verts=True,
        only_return_complete=True,
    )

    if v_samples["completed"].shape[0] == 0:
        print(
            "No vertex samples completed in this batch. Try increasing max_num_vertices."
        )
        continue
    else:
        print("Complete {} vertex samples".format(v_samples["completed"].shape[0]))

    f_samples = face_model.sample(
        v_samples,
        max_sample_length=max_num_face_indices,
        top_p=top_p_face_model,
        only_return_complete=True,
    )

    v_samples = f_samples["context"]

    if f_samples["completed"].shape[0] == 0:
        print(
            "No face samples completed in this batch. Try increasing max_num_face_indices."
        )
        continue
    else:
        print("complete {} face samples".format(f_samples["completed"].shape[0]))

    for k in range(f_samples["completed"].shape[0]):
        mesh_list.append(
            {
                "vertices": v_samples["vertices"][k][
                    : v_samples["num_vertices"][k].numpy()
                ].numpy(),
                "faces": data_utils.unflatten_faces(
                    f_samples["faces"][k][
                        : f_samples["num_face_indices"][k].numpy()
                    ].numpy()
                ),
            }
        )

meshes_plot = data_utils.plot_meshes(mesh_list, ax_lims=0.4)
with file_writer.as_default():
    tf.summary.image("mesh_plots", meshes_plot[None], step=0)
