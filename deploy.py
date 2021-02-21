# goes thru entire pipeline

from utils.model import Conv_Model
import torch
# from myargs import args
import time
import numpy as np
from scipy import signal


class DeployModel:

    def __init__(self, model_path, channels=8, seq_length=500):

        self.ch = channels
        self.seq_len = seq_length

    def data_gen_test(self):
        # Placeholder dummy data
        data = np.random.normal(loc=0, scale=1, size=(self.ch, self.seq_len))
        return data

    def get_data_and_model(self):
        """
        Function which returns data given to the model and the model output
        in the form (data, model_output)
        Return: 
            - data: List[List[float]] - an (8, 500) array of floats.
                    Each row represents a channel, and column a snapshot in time
            - output: int - 0 or 1, corresponding to the model's prediction
                            of whether the user is providing an ON or OFF signal
        
        """
        # get generator
        x = self.data_gen_test()
        # no grad for eval
        with torch.no_grad():
            # send data and get model output
            proc_data = self.process_data(x)
            out = self.model(proc_data)
            out = torch.argmax(out, dim=1).item()
            data = [d.tolist() for d in x]
        return data, out