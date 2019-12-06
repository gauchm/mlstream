"""
Large parts of this implementation are taken over from
https://github.com/kratzert/ealstm_regional_modeling.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from ..datasets import LumpedBasin, LumpedH5
from .base_models import LumpedModel
from .nseloss import NSELoss


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LumpedLSTM(LumpedModel):
    """(EA-)LSTM model for lumped data. """

    def __init__(self, num_dynamic_vars: int, num_static_vars: int, use_mse: bool = True,
                 no_static: bool = False, concat_static: bool = False,
                 run_dir: Path = None, n_jobs: int = 1, hidden_size: int = 256,
                 learning_rate: float = 1e-3, learning_rates: Dict = {}, epochs: int = 30,
                 initial_forget_bias: int = 5, dropout: float = 0.0, batch_size: int = 256,
                 clip_norm: bool = True, clip_value: float = 1.0):
        input_size_stat = 0 if no_static else num_static_vars
        input_size_dyn = num_dynamic_vars if (no_static or not concat_static) \
            else num_static_vars + num_dynamic_vars

        self.no_static = no_static
        self.run_dir = run_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rates = learning_rates
        self.learning_rates[0] = learning_rate
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.n_jobs = n_jobs
        self.use_mse = use_mse

        self.model = Model(input_size_dyn=input_size_dyn,
                           input_size_stat=input_size_stat,
                           concat_static=concat_static,
                           no_static=no_static,
                           hidden_size=hidden_size,
                           initial_forget_bias=initial_forget_bias,
                           dropout=dropout).to(DEVICE)
        self.loss_func = nn.MSELoss() if use_mse else NSELoss()

    def load(self, model_file: Path) -> None:
        self.model.load_state_dict(torch.load(model_file, map_location=DEVICE))

    def train(self, ds: LumpedH5) -> None:

        val_indices = np.random.choice(len(ds), size=int(0.1 * len(ds)), replace=False)
        train_indices = [i for i in range(len(ds)) if i not in val_indices]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        self.train_loader = DataLoader(ds, self.batch_size,
                                       sampler=train_sampler,
                                       drop_last=False,
                                       num_workers=self.n_jobs)
        self.val_loader = DataLoader(ds, self.batch_size,
                                     sampler=val_sampler,
                                     drop_last=False,
                                     num_workers=self.n_jobs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rates[0])

        for epoch in range(1, self.epochs + 1):
            # set new learning rate
            if epoch in self.learning_rates.keys():
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rates[epoch]

            self._train_epoch(epoch)
            val_loss = self._val_epoch()
            print(f"# Epoch {epoch}: validation loss: {val_loss:.7f}.")

            model_path = self.run_dir / f"model_epoch{epoch}.pt"
            torch.save(self.model.state_dict(), str(model_path))

        print(f"Model saved as {model_path}.")

    def predict(self, ds: LumpedBasin) -> np.ndarray:
        self.model.eval()

        loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)
        preds, obs = None, None
        with torch.no_grad():
            for data in loader:
                if len(data) == 2:
                    x, y = data
                    x = x.to(DEVICE)
                    p = self.model(x)[0]
                elif len(data) == 3:
                    x_d, x_s, y = data
                    x_d, x_s = x_d.to(DEVICE), x_s.to(DEVICE)
                    p = self.model(x_d, x_s[:, 0, :])[0]

                if preds is None:
                    preds = p.detach().cpu()
                    obs = y
                else:
                    preds = torch.cat((preds, p.detach().cpu()), 0)
                    obs = torch.cat((obs, y), 0)

        return preds.numpy(), obs.numpy()

    def _train_epoch(self, epoch: int):
        """Trains model for a single epoch.

        Parameters
        ----------
        epoch : int
            Current Number of epoch
        """
        self.model.train()

        # process bar handle
        pbar = tqdm(self.train_loader, file=sys.stdout)
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        running_loss = 0
        for i, data in enumerate(pbar):
            # delete old gradients
            self.optimizer.zero_grad()

            # forward pass through LSTM
            if len(data) == 3:
                x, y, q_stds = data
                x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
                predictions = self.model(x)[0]

            # forward pass through EALSTM
            elif len(data) == 4:
                x_d, x_s, y, q_stds = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                predictions = self.model(x_d, x_s[:, 0, :])[0]

            # MSELoss
            if self.use_mse:
                loss = self.loss_func(predictions, y)

            # NSELoss needs std of each basin for each sample
            else:
                q_stds = q_stds.to(DEVICE)
                loss = self.loss_func(predictions, y, q_stds)

            # calculate gradients
            loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)

            # perform parameter update
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix_str(f"Loss: {loss.item():.6f} / Mean: {running_loss / (i+1):.6f}")

    def _val_epoch(self) -> float:
        """Calculates loss on validation set during training.

        Returns
        -------
        loss : float
            Mean validation loss
        """
        self.model.eval()

        loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                # forward pass through LSTM
                if len(data) == 3:
                    x, y, q_stds = data
                    x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
                    predictions = self.model(x)[0]

                # forward pass through EALSTM
                elif len(data) == 4:
                    x_d, x_s, y, q_stds = data
                    x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                    predictions = self.model(x_d, x_s[:, 0, :])[0]

                # MSELoss
                if self.use_mse:
                    loss += self.loss_func(predictions, y).item()

                # NSELoss needs std of each basin for each sample
                else:
                    q_stds = q_stds.to(DEVICE)
                    loss += self.loss_func(predictions, y, q_stds).item()

        return loss / len(self.val_loader)


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initializes the model.

        Parameters
        ----------
        input_size_dyn : int
            Number of dynamic input features.
        input_size_stat : int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size : int
            Number of LSTM cells/hidden units.
        initial_forget_bias : int
            Value of the initial forget gate bias. (default: 5)
        dropout : float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static : bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static : bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None
        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


class EALSTM(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)

    Model details: https://arxiv.org/abs/1907.08456

    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0

    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_stat, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass on the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh)
                     + torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class LSTM(nn.Module):
    """Implementation of the standard LSTM.


    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes all learnable parameters of the LSTM. """
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data
        nn.init.constant_(self.bias.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass on the model.

        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))
            f, i, o, g = gates.chunk(4, 1)

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n
