import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size=37, hidden_size=16, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return output, (hidden, cell)
    
## 디코더
class Decoder(nn.Module):

    def __init__(self, input_size=37, hidden_size=16, output_size=37, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.1, bidirectional=False)

        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        output, (hidden, cell) = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        prediction = self.fc(output)

        return prediction, (hidden, cell)
    
## LSTM Auto Encoder
class LSTMAutoEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 **kwargs) -> None:
        """
        :param input_dim: 변수 Tag 갯수
        :param hidden_dim: 최종 압축할 차원 크기
        :param kwargs:
        """

        super(LSTMAutoEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        if "num_layers" in kwargs:
            num_layers = kwargs.pop("num_layers")
        else:
            num_layers = 1

        self.encoder = Encoder(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_size=input_dim,
            output_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self,x):
        batch_size, sequence_length, var_length = x.size()

        _,encoder_hidden = self.encoder(x)
        hidden = encoder_hidden
        temp_input = torch.zeros((batch_size,1,var_length),dtype=torch.float).to(x.device)
        reconstruct_output=[]
        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input,hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output,dim=1)
        
        return reconstruct_output
