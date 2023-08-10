from helpers.model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model.")
    parser.add_argument("-nt", dest="preload", action="store_true", help="Run without retraining")
    parser.add_argument("-ns", dest="save", action="store_false", help="Do not save model")
    parser.add_argument("-d", dest="dir", type=str, default=OUTPUT_DIR, help="Directory of model")
    parser.add_argument("-out", dest="outname", type=str, default=None, help="Name of output model")
    parser.add_argument(
        "-r", dest="event_range", type=tuple[int, int], default=EVENT_RANGE, help="Range of events for training"  # type: ignore
    )
    parser.add_argument("-e", dest="event_range", type=tuple[int, int, int], default=EPOCHS, help="Epochs for training")  # type: ignore
    parser.add_argument(
        "-eh",
        dest="event_range",
        type=tuple[int, int, int],  # type: ignore
        default=EPOCHS_HARD,
        help="Epochs for hard negative training",
    )
    parser.add_argument(
        "-continue", dest="continue_train", action="store_true", help="Continue training from last model"
    )

    args = parser.parse_args()
    kwargs = vars(args)

    if isinstance(kwargs["event_range"], tuple):
        kwargs["event_range"] = range(*kwargs["event_range"])

    get_logger(tag="train_model").debug(f"Vars: { kwargs}")

    model = get_model(**kwargs)
