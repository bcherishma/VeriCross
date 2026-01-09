import logging
from loguru import logger

def setup_logging():
    logger.add("logs/vericross.log", rotation="10 MB")
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelname, record.getMessage())
    logging.getLogger().addHandler(InterceptHandler())