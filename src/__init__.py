import logging
import os
import typing as T
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


if not os.getenv("PRINT_LOGS") or os.environ["PRINT_LOGS"] == "0":
    print("not printing logs")
    PRINT_LOGS = False
else:
    print("printing logs")
    PRINT_LOGS = True

log_name = os.getenv("LOG_NAME")


class LogfireDummy:
    def __init__(self):
        self.logger = logging.getLogger("LogfireDummy")
        self.logger.setLevel(logging.DEBUG)
        
        # Store the timestamp when the logger is initialized
        self.start_time = datetime.now()

        self.start_date = self.start_time.strftime("%Y-%m-%d")

        os.makedirs(f"logs/{self.start_date}", exist_ok=True)
        self.file_handler = logging.FileHandler(f"logs/{self.start_date}/{log_name}.log")
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)


    def debug(
        self,
        msg_template: str,
        /,
        **attributes: T.Any,
    ) -> None:
        try:
            s = msg_template
            if attributes:
                s = f"{s}\n**{attributes}**\n"
            if PRINT_LOGS:
                msg = f"LOGGER: {s}"
                print(msg)
                self.logger.debug(s)

            else:
                if "KAGGLE" not in os.environ:
                    self.logger.debug(s)
                else:
                    # make sure no non-sdk logs are recorded
                    if "anthropic" in s or "Transform" in s or "limit" in s:
                        self.logger.debug(s)
        except Exception:
            pass


if os.getenv("LOGFIRE_TOKEN"):
    import logfire

    logfire.configure(inspect_arguments=True)
else:
    logfire = LogfireDummy()


if not os.getenv("PLOT") or os.environ["PLOT"] == "0":
    PLOT = False
else:
    PLOT = True

if os.environ.get("USE_GRID_URL", "0") == "0":
    USE_GRID_URL = False
else:
    USE_GRID_URL = True
