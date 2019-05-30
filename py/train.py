from keras.callbacks import *
from data_generators import *
from models import get_model


def schedule(epoch):
    if epoch < 250:
        return 1e-3
    elif epoch < 500:
        return 1e-4
    else:
        return 1e-5


if __name__ == "__main__":
    data_csv = "../data/annotations/data_iso.csv"
    data_validation_csv = "../data/annotations/data_iso_val.csv"
    path_to_data = "../data/data_iso/"
    path_to_val_data = "../data/data_iso_val/"

    model_path = "../models/cnn_isometry_100k_2.h5"

    train_generator = data_gen_iso(data_csv, path_to_data,  256)
    val_generator = data_gen_iso(data_validation_csv, path_to_val_data, 64, False, "_val")

    schd = LearningRateScheduler(schedule)
    ckpt = ModelCheckpoint(model_path,
                           monitor="val_loss", mode="min",
                           save_best_only=True)
    tboard = TensorBoard()
    # m = get_model(outputs=(8, 4))
    m = get_model(outputs=8)
    m.fit_generator(train_generator, epochs=2000, callbacks=[schd, ckpt, tboard],
                    steps_per_epoch=1024, validation_data=val_generator, validation_steps=80)