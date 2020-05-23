from skopt.space import Real, Integer, Categorical
import signal
import io
import time
import subprocess
import sys
import pathlib
import os

from train_instance import TrainInstance
sys.path.append("HyperParameter-Optimizer/")
from gaussian_process import GaussianProcessSearch

file_dir = str(pathlib.Path(__file__).resolve().parents[0])


def signal_handler(sig, frame):
    print('\nSIGINT signal received: killing instances...')
    for instance in instances:
        instance.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

search_space = [
    Real(low=0.1, high=1., name='reproduction_reward'),
    Real(low=1e-7, high=1e-4, name='time_step_modifier'),
    Real(low=0., high=1e-5, name='pop_reward_modifier'),
]

if __name__ == "__main__":
    num_instances = 3

    # Paths and files
    gpro_input_file = None  # Use None to start from zero
    env_dir = "envs/"
    env_path = os.path.join(env_dir, "optimization_test.x86_64")
    log_files_dir = os.path.join(file_dir, "logs/")
    output_files_dir = "out_files/"
    config_file = "config/chickens.yaml"

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space={},
                                      evaluator=None,
                                      input_file=gpro_input_file,
                                      output_file='results.csv')

    # Instantiate training instances
    instances = []
    for i in range(num_instances):
        instances.append(TrainInstance(port=i,
                                       env_path=env_path,
                                       log_files_dir=log_files_dir,
                                       output_files_dir=output_files_dir,
                                       config_file=config_file))

    # Start training all instances
    candidates = []
    if gpro_input_file is None:  # If first points, sample random
        candidates = gp_search.get_random_candidate(num_instances)
    else:
        candidates = gp_search.get_next_candidate(num_instances)
    for i in range(num_instances):
        instances[i].train(candidates[i])

    while(True):
        time.sleep(5)  # refresh rate in seconds
        for i in range(num_instances):
            instance = instances[i]
            if instance.inactive:
                candidate = gp_search.get_next_candidate(1)[0]
                instance.train(candidate)
            elif instance.is_done():
                instance_params = instance.meta_params
                instance_result = instance.get_val()
                print("instance_params:", instance_params)
                print("instance_result:", instance_result)
                gp_search.add_point_value(instance_params, instance_result)
                gp_search.save_values()
                instance.kill()
            # print("Instance", i, "has value:", instance.get_val())
