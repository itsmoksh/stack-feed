import logging

def setup_logger(name, log_file='stack_feed.log', level=logging.DEBUG):
    # Create a custom logger
    logger = logging.getLogger(name)

    # Configure the custom logger
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    return logger