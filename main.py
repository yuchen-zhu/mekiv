import yaml
import logging
from typing import Optional
import click
from pathlib import Path
import uuid
from shutil import copyfile, make_archive
import os
from os import PathLike
import datetime


from miv.utils.custom_logging import configure_logger
from miv.experiments import experiments


DUMP_DIR = Path.cwd().joinpath('dumps')
SRC_DIR = Path.cwd().joinpath('miv')

SLACK_URL = None
NUM_GPU = None
if Path.cwd().joinpath('miv/config.yaml').exists():
    SLACK_URL = yaml.load(Path.cwd().joinpath('miv/config.yaml').open('r'))['slack']
    NUM_GPU = yaml.load(Path.cwd().joinpath('miv/config.yaml').open('r')).get('num_gpu', None)

SCRIPT_NAME = Path(__file__).stem
LOG_DIR = Path.cwd().joinpath(f'logs/{SCRIPT_NAME}')

logger = logging.getLogger()


@click.group()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--debug/--release', default=False)
@click.pass_context
def main(ctx, config_path, debug):
    if(debug):
        # Change logging level to debug
        logger.setLevel(logging.DEBUG)
        logger.handlers[-1].setLevel(logging.DEBUG)
        logger.debug("debug")

    foldername = str(datetime.datetime.now().strftime("%m-%d-%H-%M-%S"))
    dump_dir = DUMP_DIR.joinpath(foldername)
    os.mkdir(dump_dir)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ctx.obj["dump_dir"] = dump_dir
    ctx.obj["config"] = config
    yaml.dump(config, open(dump_dir.joinpath("configs.yaml"), "w"))
    make_archive(dump_dir.joinpath("src"), "zip", root_dir=SRC_DIR)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def merrorkiv(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("merrorkiv")
    os.mkdir(dump_dir)
    experiments("MerrorKIV", config, dump_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def lvm(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("lvm")
    os.mkdir(dump_dir)
    experiments("LVM", config, dump_dir, num_thread, NUM_GPU)


# @main.command()
# @click.pass_context
# @click.option("--num_thread", "-t", default=1, type=int)
# def oraclekiv(ctx, num_thread):
#     config = ctx.obj["config"]
#     dump_dir = ctx.obj["dump_dir"]
#     dump_dir = dump_dir.joinpath("oraclekiv")
#     os.mkdir(dump_dir)
#     experiments("oracle_KIV", config, dump_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def kivoracle(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("kivoracle")
    os.mkdir(dump_dir)
    experiments("KIV_oracle", config, dump_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def kivm(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("kivm")
    os.mkdir(dump_dir)
    experiments("KIV_M", config, dump_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def kivn(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("kivn")
    os.mkdir(dump_dir)
    experiments("KIV_N", config, dump_dir, num_thread, NUM_GPU)


@main.command()
@click.pass_context
@click.option("--num_thread", "-t", default=1, type=int)
def kivmn(ctx, num_thread):
    config = ctx.obj["config"]
    dump_dir = ctx.obj["dump_dir"]
    dump_dir = dump_dir.joinpath("kivmn")
    os.mkdir(dump_dir)
    experiments("KIV_MN", config, dump_dir, num_thread, NUM_GPU)


if __name__ == '__main__':
    configure_logger(SCRIPT_NAME, log_dir=LOG_DIR, webhook_url=SLACK_URL)
    try:
        main(obj={})
        logger.critical('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)