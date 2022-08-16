import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pandas as pd 
import pickle

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset',args.dataset))

    train_set = SeqDataset(dataset_folder,'train')

    test_set = SeqDataset(dataset_folder,'test')

    train_loader = DataLoader(train_set,args.batch_size,  num_workers=0,
                              shuffle=True,collate_fn=collate_fn,drop_last=True)
    test_loader = DataLoader(test_set,args.test_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)

    item_num, relation_num, entity_num = pickle.load(open(os.path.join(dataset_folder,'dataset_info.pkl'),'rb'))

    kg_set = KgDataset(dataset_folder)
    kg_loader = DataLoader(kg_set,args.batch_size, num_workers=0,
                              shuffle=True)

    entity_num = max(entity_num, kg_set.tail.max()+1)
    
    print("Load data over!")
    return train_loader, test_loader, item_num, relation_num, entity_num, kg_loader

class SeqDataset(Dataset):
    def __init__(self, datafolder, file='train') -> None:
        super().__init__()
        assert file in ['train','test'],"file only is trian or test"

        data_file = os.path.join(datafolder, file+'.pkl')
        data = pickle.load(open(data_file,'rb'))
        self.max_len = 20 # 所有数据session的最大长度

        pro_file = os.path.join(datafolder, file+'_pro.pkl')
        if os.path.exists(pro_file):
            self.item_seq, self.op_seq, self.tar, self.mask = pickle.load(open(pro_file,'rb'))
        else:
            self.item_seq = []
            self.op_seq = []
            self.tar = []
            self.mask = []
            for s in data:
                use_s = s[-self.max_len-1:]
                padding_len = max(self.max_len - len(use_s) + 1,0)
                self.item_seq.append([0]*padding_len + [i for i,o in use_s[:-1]])
                self.op_seq.append([0]*padding_len + [o for i,o in use_s[:-1]])
                self.mask.append([1]*padding_len + [0]*len(use_s[:-1]))
                self.tar.append(use_s[-1][0])
            pickle.dump((self.item_seq, self.op_seq, self.tar, self.mask), open(pro_file,'wb'))

    def __len__(self):
        return len(self.item_seq)

    def __getitem__(self, index): 
        return self.item_seq[index], self.op_seq[index], self.tar[index], self.mask[index]

class KgDataset(Dataset):
    def __init__(self, datafolder) -> None:
        super().__init__()
        kg = pd.read_csv(os.path.join(datafolder,'kg2id'))
        self.head, self.tail, self.relation = kg['head'],kg['tail'],kg['relation']
    def __len__(self):
        return len(self.head)
    def __getitem__(self, index) :
        return self.head[index], self.tail[index], self.relation[index]


def collate_fn(batch_data):

    batch_x = []
    batch_seq = []
    batch_y = []
    batch_mask = []

    batch_node = [0] # 一个batch中传入GNN的结点 
    batch_edge = [[],[]] # 对应的边 head_list, tail_list
    edge2seq = [] # seq中每个结点对应在batchnode中的下标

    for data in batch_data:
        x = data[0]
        op = data[1]
        y = data[2]
        mask = data[3]

        batch_x.append(x)
        batch_seq.append(op)
        batch_y.append(y)
        batch_mask.append(mask)

        star_index = sum(mask) # 起始下标 

        if star_index == len(x)-1: # 只有一个结点 
            batch_node.append(x[-1]) # 结点传入 
            edge2seq.append([0]*star_index + [len(batch_node) - 1]) # padding + 当前结点位置 
            batch_edge[0].append(len(batch_node) - 1) # self loop
            batch_edge[1].append(len(batch_node) - 1)
        else:
            node_set = list(set(x[star_index:]))
            now_map = dict([(i, len(batch_node)+n) for i,n in zip(node_set, range(len(node_set)))]) # 当前session item map到node
            batch_node.extend(node_set) # 当前session添加到batch

            for h,t in zip(x[star_index:-1],x[star_index+1:]):
                batch_edge[0].append(now_map[h]) # 头结点 
                batch_edge[1].append(now_map[t]) # 尾结点
            edge2seq.append([0]*star_index + [now_map[i] for i in x[star_index:]]) # padding + 当前结点位置 
            
    items = torch.LongTensor(batch_x)
    ops = torch.LongTensor(batch_seq)
    labels = torch.LongTensor(batch_y)
    masks = torch.LongTensor(batch_mask)

    nodes = torch.LongTensor(batch_node)
    edges = torch.LongTensor(batch_edge)
    edge2seq = torch.LongTensor(edge2seq)

    return items,ops,labels, masks, nodes, edges, edge2seq