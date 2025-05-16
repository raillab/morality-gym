import argparse
import sys

sys.path.append('..')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--create", type=str)
    parser.add_argument("-es", "--exec-slurms", type=str)

    parser.add_argument("-er", "--exec-run", type=str)
    parser.add_argument("-lp", "--log-path", type=str)

    parser.add_argument("-il", "--is-local", action="store_true")
    parser.add_argument("-ie", "--interactive-env", action="store_true")
    parser.add_argument("-irp", "--is-rel-path", action="store_false")

    parser.add_argument("-emr", "--exec-multi-run", type=str, nargs="+")

    parser.add_argument("-lbp", "--log-base-path", type=str)
    parser.add_argument("-ltp", "--log-temp-path", type=str)
    parser.add_argument("-lmp", "--log-multi-path", type=str, nargs="+")


    parser.add_argument("-ear", "--exec-all-runs", type=str)

    parser.add_argument("-n", "--node", type=str)
    #--node

    args = parser.parse_args()

    if args.create is not None:
        from benchmarks.benchmarker.benchmarker import Benchmarker
        benchmarker = Benchmarker(args.create, is_local=args.is_local)
        print(f"Creating run configs and slurm scripts for {args.create}...")
        benchmarker.create()
        # benchmarker.create_run_configs()
        # benchmarker.create_slurms()
    elif args.exec_slurms is not None:
        from benchmarks.benchmarker.benchmarker import Benchmarker
        if args.is_local:
            raise ValueError("Cannot execute slurm scripts locally.")
        benchmarker = Benchmarker(args.exec_slurms, is_local=False)
        benchmarker.exec_slurms()
    elif args.exec_run is not None:
        from benchmarks.benchmarker.run import exec_run
        exec_run(args.exec_run, log_path=args.log_path, is_interactive_env=args.interactive_env, node=args.node,
                 is_config_rel_path=args.is_rel_path, log_temp_path=args.log_temp_path)
    elif args.exec_multi_run is not None:
        from benchmarks.benchmarker.run import exec_multi_run
        if args.log_base_path is None or args.log_multi_path is None:
            raise ValueError("Must specify log base path and log multi paths for multi-run execution.")
        if len(args.exec_multi_run) != len(args.log_multi_path):
            raise ValueError("Must specify the same number of log relative paths as run configs.")

        exec_multi_run(args.exec_multi_run, log_base_path=args.log_base_path, log_rel_paths=args.log_multi_path,
                       node=args.node, is_config_rel_path=args.is_rel_path, log_temp_path=args.log_temp_path)
    elif args.exec_all_runs is not None:
        from benchmarks.benchmarker.run import exec_all_runs
        if args.log_base_path is None:
            raise ValueError("Must specify log base path for all-run execution.")
        exec_all_runs(args.exec_all_runs, args.log_base_path)
    else:
        raise ValueError("Exactly one of -c, -es, -er, -emr must be specified.")


if __name__ == "__main__":
    main()