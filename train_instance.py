import sys
import pathlib
import os
import subprocess
import time
import io
import yaml
import json
import signal
import copy
import numpy as np


class TrainInstance:
    def __init__(self, port, env_path, log_files_dir, output_files_dir="out_files/", config_file="config/ppo.yaml"):
        # Public
        self.__reset()
        self.env_path = str(env_path)
        self.port = port

        self.output_files_dir = str(output_files_dir)  # Where to write terminal outputs
        self.log_files_dir = str(log_files_dir)        # Where to write logs saved by environment
        self.config_file = str(config_file)            # Config file to use in training

        # Ensure directories exisits 
        pathlib.Path(self.output_files_dir).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(self.log_files_dir).mkdir(
            parents=True, exist_ok=True)

    def __reset(self):
        self.inactive = True
        self.process = None
        self.id = None

    def train(self, meta_params):
        """ Launch training instance
        """
        self.inactive = False
        self.meta_params = meta_params
        print(self.meta_params)
        self.id = self.__dict_to_string(meta_params)
        self.json_file = os.path.join(self.log_files_dir, str(self.id) + ".json")
        command = self.__get_command()
        self.output_file = os.path.join(self.output_files_dir, str(self.id) + ".out")
        log = open(self.output_file, 'a')
        self.process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
        print("Training started with PID: " + str(self.process.pid))

    def is_done(self):
        """ True if training has finished / converged / not promising
        """
        status = self.process.poll()
        if status is not None and status == 0:
            return True
        # TODO: early stopping heuristics
        return False

    def kill(self):
        """ Send SIGINT signal to process
        """
        print("Killing", self.process.pid)
        try:
            self.process.send_signal(signal.SIGINT)
        except Exception as e:
            print(e)
        self.__reset()

    def get_val(self):
        """ Return training result (for the optimizer)
        """
        last_values = self.__get_last_n_values(n=50)
        if len(last_values) > 0:
            return np.average(last_values)
        return 0

    def __get_last_n_values(self, n):
        last_values = []
        with open(self.json_file) as f:
            data = json.load(f)
            length = len(data["episodes"])
            last_episodes = data["episodes"][max(length-n, 0):]
            for episode in last_episodes:
                last_values.append(len(episode["numCreatures"]))
        return last_values

    def __get_command(self):
        # mlagents-learn config_ppo.yml --env test.x86_64 --no-graphics --train --env-args --time-step-modifier 0.0001 --reproduction-reward 0.5 --population-reward-modifier 0
        command = " mlagents-learn "
        command += str(self.config_file)
        command += " --env " + self.env_path
        command += " --run-id=" + self.id
        command += " --base-port=" + str(5000 + self.port)
        command += " --time-scale=50 --no-graphics --train --env-args "
        # Params
        command += " --time-step-modifier " + str(self.meta_params["time_step_modifier"])
        command += " --reproduction-reward " + str(self.meta_params["reproduction_reward"])
        command += " --population-reward-modifier " + str(self.meta_params["pop_reward_modifier"])
        command += " --log-file " + str(self.json_file)
        # command += str(self.id) + ".out"  # Where to save outputs
        print("command:", command)
        return command

    def __dict_to_string(self, dictionary):
        s = str(dictionary)
        s = s.replace(" ", "")
        s = s.replace("{", "")
        s = s.replace("}", "")
        s = s.replace("'", "")
        s = s.replace(",", "-")
        return s