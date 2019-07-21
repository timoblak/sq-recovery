from keras.callbacks import *
from data_generators import *
from models import get_model_rot


def schedule(epoch):
    if epoch < 250:
        return 1e-3
    elif epoch < 500:
        return 1e-4
    else:
        return 1e-5


if __name__ == "__main__":
    data_csv = "../data/annotations/data_rot.csv"
    data_val_csv = "../data/annotations/data_rot_val.csv"
    path_to_data = "../data/data/"
    path_to_val_data = "../data/data_rot_val/"

    model_path = "../models/cnn_100k_2.h5"


    train_generator = data_gen_quats(data_csv, path_to_data, 64, mode="")

    val_generator = data_gen_quats(data_val_csv, path_to_val_data, 64, mode="_rot_val")



    schd = LearningRateScheduler(schedule)
    ckpt = ModelCheckpoint(model_path,
                           monitor="val_loss", mode="min",
                           save_best_only=True)
    tboard = TensorBoard()
    m = get_model_rot(outputs=(8, 4))

    m.fit_generator(train_generator, epochs=2000,
                    callbacks=[schd, ckpt, tboard],
                    steps_per_epoch=1500,
                    validation_data=val_generator,
                    validation_steps=80)