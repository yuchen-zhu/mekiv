import datetime
import logging
from pathlib import Path
from shutil import make_archive

import click
import yaml

from miv.experiments import experiments
from miv.utils import configure_logger

CWD = Path.cwd()
DUMP_DIR = CWD / "dumps"
SRC_DIR = CWD / "miv"
GLOBAL_CONFIG_PATH = CWD / "miv" / "config.yaml"

# for sending message to slack
SLACK_URL = None
NUM_GPU = None
if GLOBAL_CONFIG_PATH.exists():
    global_config = yaml.load(
        GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"), Loader=yaml.SafeLoader
    )
    SLACK_URL = global_config.get("slack")
    NUM_GPU = global_config.get("num_gpu")

SCRIPT_NAME = Path(__file__).stem
LOG_DIR = CWD / "logs" / SCRIPT_NAME

logger = logging.getLogger("mekiv")


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.argument("method", type=str)
@click.option("--debug/--release", default=False)
@click.option("--num_thread", default=1)
def main(config_path: Path, method: str, debug: bool, num_thread: int):
    configure_logger(log_dir=LOG_DIR, webhook_url=SLACK_URL)
    print(config_path, method, debug, num_thread)
    if debug:
        # Change logging level to debug
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("debug")

    foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    try:
        dump_dir = DUMP_DIR / foldername
        dump_dir.mkdir(parents=True, exist_ok=True)
        config = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader
        )

        (dump_dir / "configs.yaml").write_text(yaml.dump(config), encoding="utf-8")
        make_archive(dump_dir / "src", "zip", root_dir=SRC_DIR)
        if method not in {"MerrorKIV", "KIV_oracle", "KIV_M", "KIV_N", "KIV_MN"}:
            raise ValueError(
                "Choose Method from the following: 'MerrorKIV', 'KIV_oracle', 'KIV_M', 'KIV_N', 'KIV_MN'"
            )
        else:
            method_dump_dir = dump_dir / method
            method_dump_dir.mkdir()
            experiments(
                alg_name=method,
                configs=config,
                dump_dir=method_dump_dir,
                num_cpus=num_thread,
                num_gpus=NUM_GPU,
            )
        logger.info("===== Script completed successfully! =====")
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
