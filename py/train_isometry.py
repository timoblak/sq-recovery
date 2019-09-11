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
    data_csv = parse_csv("../data/annotations/data_iso.csv")
    data_val_csv = parse_csv("../data/annotations/data_iso_val.csv")
    path_to_data = "../data/data_iso/"
    path_to_val_data = "../data/data_iso_val/"

    model_path = "../models/cnn_isometry_100k_2.h5"

    BATCH_SIZE = 1
    # Using test db to train for testing purposes
    # NB_TRAIN = len(data_csv)
    NB_TRAIN = len(data_csv)
    NB_TEST = len(data_val_csv)
    train_generator = data_gen_iso(data_csv, path_to_data,  256)
    val_generator = data_gen_iso(data_val_csv, path_to_val_data, 64)

    schd = LearningRateScheduler(schedule)
    ckpt = ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True)
    tboard = TensorBoard()

    m = get_model(outputs=8)
    m.fit_generator(train_generator, epochs=2000,
                    callbacks=[schd, ckpt, tboard],
                    steps_per_epoch=NB_TRAIN / BATCH_SIZE,
                    validation_data=val_generator,
                    validation_steps=NB_TEST / BATCH_SIZE)