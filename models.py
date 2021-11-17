### set up encoder & decoder networks
import torch
import torch.nn as nn


class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=2, bidirectional=True,
                 batch_first=True):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=self.batch_first)

    def forward(self, x_input, hidden):
        lstm_out, hidden = self.lstm(x_input, hidden)
        return lstm_out, hidden

    def init_hidden(self, device):
        return (
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device),
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device))


class lstm_conductor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=True,
                 batch_first=True):
        super(lstm_conductor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=self.batch_first)
        self.linear = nn.Linear((self.bidirectional + 1) * self.hidden_size, self.output_size)
        # self.tanh = nn.Tanh()
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x_input, hidden):
        lstm_out, hidden = self.lstm(x_input, hidden)
        lstm_out = self.linear(lstm_out.squeeze(0))
        # print(lstm_out)
        # lstm_out = self.tanh(lstm_out)
        # lstm_out = self.softmax(lstm_out)
        # print(lstm_out)
        # lstm_out = self.softmax(lstm_out)
        # return lstm_out, hidden
        return lstm_out, hidden

    def init_hidden(self, device):
        return (
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device),
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device))


class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers=1, bidirectional=True,
                 batch_first=True):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=self.batch_first)
        self.linear = nn.Linear((self.bidirectional + 1) * self.hidden_size, self.output_size)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x_input, hidden):
        lstm_out, hidden = self.lstm(x_input, hidden)
        lstm_out = self.linear(lstm_out.squeeze(0))
        # lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

    def init_hidden(self, device):
        return (
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device),
        torch.zeros((self.bidirectional + 1) * self.num_layers, self.batch_size, self.hidden_size, device=device))

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, quantized.contiguous(), perplexity, encodings