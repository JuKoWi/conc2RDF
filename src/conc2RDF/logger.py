import logging


def set_up_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="report.log",
        filemode="w",
    )
