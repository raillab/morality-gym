import argparse
import os
import sys

def _join_paths(paths):
    return os.path.normpath(os.path.join(*paths))


def _comp_base_path(path, level=1):
    curr_path = path
    for i in range(level):
        curr_path = os.path.normpath(os.path.dirname(curr_path))

    curr_path = os.path.normpath(curr_path)

    return curr_path

PROJECT_ROOT = _comp_base_path(os.path.abspath(__file__), level=2)
# print(PROJECT_ROOT)

sys.path.append(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--create", type=str)
    parser.add_argument("-es", "--exec-slurms", type=str)


    parser.add_argument("-er", "--exec-run", type=str)
    parser.add_argument("-lp", "--log-path", type=str)
    parser.add_argument("-mp", "--model-path", type=str)

    parser.add_argument("-emr", "--exec-multi-run", type=str, nargs="+")

    parser.add_argument("-lmp", "--log-multi-path", type=str, nargs="+")
    # parser.add_argument("-mmp", "--model-multi-path", type=str, nargs="+")

    parser.add_argument("-n", "--node", type=str)
    parser.add_argument("-ie", "--interactive-env", action="store_true")

    args = parser.parse_args()

    # CREATE #
    if args.create is not None:
        from baselines.benchmarker.benchmarker import Benchmarker
        benchmarker = Benchmarker(args.create)
        print(f">>> Creating run configs and slurm scripts for {args.create}...")
        benchmarker.create()
    # EXEC SLURMS #
    elif args.exec_slurms is not None:
        from baselines.benchmarker.benchmarker import Benchmarker
        # from experiments.benchmarker.benchmarker import Benchmarker
        benchmarker = Benchmarker(args.exec_slurms)
        print(f"\n>>> Executing slurms for {args.exec_slurms}...\n")
        benchmarker.exec_slurms()
    # EXEC RUN #
    elif args.exec_run is not None:
        from baselines.benchmarker.run import run

        print(f"\n>>> Executing run {args.exec_run}...\n")
        run(args.exec_run, node=args.node, log_dir=args.log_path, is_interactive=args.interactive_env)
    # EXEC MULTI RUN #
    elif args.exec_multi_run is not None:
        from baselines.benchmarker.run import run
        if args.log_multi_path is None:
            raise ValueError(f"--log-multi-path must be specified when using --exec-multi-run")


        if len(args.exec_multi_run) != len(args.log_multi_path):
            raise ValueError(f"--exec-multi-run and --log-multi-path must be the same length")

        n_runs = len(args.exec_multi_run)
        for i in range(n_runs):
            print(f"\n>>> Executing run [{i+1}/{n_runs}] - {args.exec_multi_run[i]}...\n")
            run(
                config=args.exec_multi_run[i],
                node=args.node,
                log_dir=args.log_multi_path[i]
            )
    # INTERACTIVE ENV #
    elif args.interactive_env is not None:
        raise NotImplementedError
        # from experiments.benchmarker.run import run
        # print(f"\n>>> Interactive env for {args.interactive_env}...\n")
        # run(args.interactive_env, is_interactive=True)
    else:
        raise ValueError(f"Flags not specified.")


if __name__ == "__main__":
    main()