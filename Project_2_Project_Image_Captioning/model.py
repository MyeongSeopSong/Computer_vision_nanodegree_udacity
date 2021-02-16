import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.drop_prob = 0.5
        self.n_layers = num_layers
        self.n_hidden = hidden_size
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout=self.drop_prob, batch_first=True)        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(self.drop_prob)        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        ##embeding captions
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, features, captions):
        #print(captions.shape)
        cap_embedding=self.embed(captions[:,:-1])
        #print(features.shape)
        #print(cap_embedding.shape)
        embedded=torch.cat((features.unsqueeze(1), cap_embedding), 1)
        outputs, states = self.lstm(embedded)
        outputs = self.dropout(outputs)
        scores = self.fc(outputs) 
        return scores
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #print(inputs.shape)
        outputs, states = self.lstm(inputs)
        outputs = self.dropout(outputs)
        scores = self.fc(outputs) 
        #print(scores.shape)
        _, predicted_ids = torch.max(scores, 2)
        #print(predicted_ids.item())
        ids_list=[]
        ids_list.append(predicted_ids.item())
        # Now pass in the previous character and get a new one
        for ii in range(max_len-1):
            cap_embedding=self.embed(predicted_ids)
            #print(cap_embedding.shape)
            outputs, states = self.lstm(cap_embedding,states)
            outputs = self.dropout(outputs)
            scores = self.fc(outputs) 
            _, predicted_ids = torch.max(scores, 2)
            ids_list.append(predicted_ids.item())
            if predicted_ids.item()==1:
                return  ids_list
        return  ids_list