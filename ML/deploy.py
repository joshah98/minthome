from ML.model import Conv_Model
import torch
import time
import numpy as np
from scipy import signal


class DeployModel:

    def __init__(self, model_path, channels=8, seq_length=500):

        self.ch = channels
        self.seq_len = seq_length

        # load model
        model = Conv_Model()
        self.model = model.eval()

    def process_data(self, data, samp_freq=250, nperseg=32):
        # assume we receive ch*seq_len array, choose nperseg to make a square array
        spectro = []
        for cha in range(self.ch):
            # use n per seg of 32 to have a shape of 17x17
            f, t, spectrogram = signal.spectrogram(data[cha], fs=samp_freq, nperseg=nperseg)
            spectro.append(spectrogram)

        # finally convert to array
        spectro = np.asarray(spectro)

        # put spec between 0 and 1
        spectro = spectro - spectro.min()
        spectro = spectro / spectro.max()

        spectro = np.ascontiguousarray(spectro)
        spectro = torch.from_numpy(spectro).float().unsqueeze(0)
        return spectro

    def data_gen_test(self):
        """
        Generator which generates dummy data of format (channels, sequence_length)
        simulating the headset
        Returns: data - np.array simulating headset data
        """
        data = np.random.normal(loc=0, scale=1, size=(self.ch, self.seq_len))
        return data

    def get_data_and_model(self):
        """
        Function which returns data fed into model and the model output
        in a tuple of the form (data, model_output)
        Returns: 
          - data: np.array of floats, size (channels, sequence_length)
          - model output: int, either a 0 or a 1 depending on what the ML model predicts
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
