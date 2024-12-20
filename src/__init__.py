import logging
import os
import typing as T
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


if not os.getenv("PRINT_LOGS") or os.environ["PRINT_LOGS"] == "0":
    print("not printing logs")
    PRINT_LOGS = False
else:
    print("printing logs")
    PRINT_LOGS = True




class LogfireDummy:
    def __init__(self):
        self.logger = logging.getLogger("LogfireDummy")
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory with date
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = Path("./logs") / current_date
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%H-%M-%S")
        log_filename = f"log_{timestamp}.log"
        log_path = self.log_dir / log_filename
        
        # Ensure the logger doesn't already have handlers
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, mode='a')
            formatter = logging.Formatter(
                "%(asctime)s EST: %(message)s", 
                datefmt="%Y-%m-%d %I:%M:%S %p"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Print where logs will be saved
            print(f"Logs will be saved to: {log_path}")

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
            
            # Always write to file, optionally print
            self.logger.debug(s)
            
            if PRINT_LOGS:
                print(f"LOGGER: {s}")
                
        except Exception as e:
            print(f"Logging error: {e}")


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



