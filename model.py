import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from bonfire.model import ModelTemplate
from bonfire.layers.embeddings import Embedding


class DocumentClassifier(ModelTemplate):
    def __init__(self, config=None):
        super(DocumentClassifier, self).__init__(config)
        self.vocabulary = None
        self.pretrained_embeddings = None
        self.loss_fn = None
        self.optimizer = None
        self.is_built = False

    def initialize_features(self, data=None):
        assert isinstance(data, Dataset), \
            "data should be of type Dataset, " \
            "got {} instead".format(type(data))

        self.vocabulary = data.vocabulary
        self.initialized = True

    def build_model(self):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def update(self, input, output):
        self.train()

        input = Variable(input)
        tags = Variable(output)

        if self.gpu_device is not None:
            input = input.cuda()
            tags = tags.cuda()

        self.optimizer.zero_grad()  # Initialize Gradients to Zero
        out = self.forward(input)  # Forward Pass
        loss = self.loss_fn(out, tags)  # Compute Loss
        loss.backward()  # Backward Pass
        self.optimizer.step()  # optimizer Step

        return loss.data[0]

    def predict(self, input):
        self.eval()

        input = Variable(input)

        if self.gpu_device is not None:
            input = input.cuda()

        out = self.forward(input)  # Forward Pass
        scores = torch.exp(out).data.cpu().numpy()[:, 1].tolist()

        return scores


class MLP(DocumentClassifier):
    def __init__(self, config):
        super(MLP, self).__init__(config)

        self.embedding_size = self.config['embedding_size']
        self.hidden_size = self.config['hidden_size']

        self.emb = None
        self.linear = None
        self.linear_out = None
        self.activation = None

    def build_model(self):
        self.emb = Embedding(len(self.vocabulary), self.embedding_size)
        self.linear = nn.Linear(self.config, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, 1)
        self.activation = F.tanh

    def forward(self, x):
        h = self.emb(x)
        h = h.mean(dim=1)
        h = self.activation(self.linear(h))
        h = self.linear_out(h)
        p = F.log_softmax(h, dim=-1)
        return p
