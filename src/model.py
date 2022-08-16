import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv


class MKMSR(nn.Module):
    def __init__(self,args, item_num, relation_num, entity_num) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_item = item_num
        self.n_entity = entity_num
        self.n_relation = relation_num

        self.entity_embedding = nn.Embedding(self.n_entity, self.hidden_size, padding_idx=0)
        self.op_embedding = nn.Embedding(100, self.hidden_size, padding_idx=0)
        

        # recommendation
        self.gnn = GatedGraphConv(self.hidden_size, args.layer)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.linear_one = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_two = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_three = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)
        self.prediction_mlp = nn.Linear(self.hidden_size*2, 1)

        # kg
        self.relation_embedding = nn.Embedding(self.n_relation, self.hidden_size)
        self.d_r = nn.Embedding(self.n_relation, self.hidden_size)
        self.kg_loss = nn.MSELoss()

        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.05
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, nodes, edges, edge2seq, ops, mask):
        
        gnn_out = self.gnn(self.entity_embedding(nodes), edges)[edge2seq]
        gru_out,_ = self.gru(self.op_embedding(ops))
        final_emb = torch.concat([gnn_out,gru_out],dim=-1)
        alpha = self.linear_three(torch.sigmoid(self.linear_one(final_emb[:,-1]).unsqueeze(1) + self.linear_two(final_emb)))
        alpha[mask] = 0
        sg = torch.sum(final_emb*alpha, dim=1)
        s = self.linear_transform(torch.concat([final_emb[:,-1], sg],dim=-1))
        item_emb = self.entity_embedding.weight[:self.n_item]

        prob = torch.matmul(s, item_emb.T)
        return prob

    def kg_train(self, h, t, r):
        w_r = self.relation_embedding(r)
        d_r = self.d_r(r)

        i = self.entity_embedding(h)
        a = self.entity_embedding(t)

        loss = torch.norm((i - (w_r @ i.T @ w_r)) + d_r + (a - (w_r @ a.T @ w_r)),2)

        return loss