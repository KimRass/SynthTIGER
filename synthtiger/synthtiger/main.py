"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
from pprint import pprint
import time

import synthtiger


def run(args):
    if args.config is not None:
        config = synthtiger.read_config(args.config)

    pprint(config)

    synthtiger.set_global_random_seed(args.seed)
    template = synthtiger.read_template(path=args.template_path, name=args.template_name, config=config)
    generator = synthtiger.generator(
        path=args.template_path,
        name=args.template_name,
        config=config,
        count=args.count,
        worker=args.n_workers,
        seed=args.seed,
        retry=True,
        verbose=args.verbose,
    )

    if args.output is not None:
        template.init_save(args.output)

    for idx, (task_idx, data) in enumerate(generator):
        if args.output is not None:
            template.save(args.output, data, task_idx)
        print(f"Generated {idx + 1} data (task {task_idx})")

    if args.output is not None:
        template.end_save(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", help="template_path file path.")
    parser.add_argument("--template_name", help="Template class name.")
    parser.add_argument("--config", help="Config file path.")
    parser.add_argument("--output", help="Directory path to save data.")
    parser.add_argument("--count", type=int, default=100, help="Number of output data. [default: 100]")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers. If 0, It generates data in the main process. [default: 0]"
    )
    parser.add_argument("--seed", type=int, help="Random seed. [default: None]")
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print error messages while generating data."
    )
    args = parser.parse_args()

    pprint(vars(args))
    return args


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
