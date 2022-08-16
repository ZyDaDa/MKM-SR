from tqdm import tqdm
from dataset import load_data

import torch
import time
from parse import get_parse
from utils import fix_seed, topk_mrr_hr
from model import MKMSR

if __name__ == '__main__':

    args = get_parse()

    fix_seed()
    start_time = time.time() # 开始时间 

    train_loader, test_loader, item_num, relation_num, entity_num, kg_loader = load_data(args)


    model = MKMSR(args, item_num, relation_num, entity_num)
    model.to(args.device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    e_stop = 0
    hr_max,mrr_max = {k:0 for k in args.topk}, {k:0 for k in args.topk} # 记录最高的topk分数
    for e in range(args.epoch):
        epo_start_time = time.time()
        # train kg
        bar = tqdm(kg_loader, total=len(kg_loader),ncols=100)
        for head, tail, relation in bar:
            loss = model.kg_train(head.to(args.device), 
                                tail.to(args.device),
                                relation.to(args.device))*args.kg_loss_rate
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            bar.set_postfix(Epoch=e)
        # rec train
        model.train()
        all_loss = 0.0
        bar = tqdm(train_loader, total=len(train_loader),ncols=100)
        for items,ops,labels, masks, nodes, edges, edge2seq in bar:
            output = model(nodes.to(args.device), 
                            edges.to(args.device),
                            edge2seq.to(args.device),
                            ops.to(args.device),
                            masks.to(args.device))
            optimizer.zero_grad() 
            loss = model.loss_function(output, labels.to(args.device)) # 因为没有测试pad
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            bar.set_postfix(Epoch=e)
        scheduler.step()
        print("train tiem %d s"%(time.time()-epo_start_time))

        # 测试模型
        model.eval()
        all_loss = 0.0
        hr, mrr = dict((k,0.0) for k in args.topk), dict((k,0.0) for k in args.topk) # 当前epoch每个指标的分数
        test_num = 0
        for items,ops,labels, masks, nodes, edges, edge2seq in test_loader:
            output = model(nodes.to(args.device), 
                            edges.to(args.device),
                            edge2seq.to(args.device),
                            ops.to(args.device),
                            masks.to(args.device))
            for k in hr.keys():
                this_hr, this_mrr = topk_mrr_hr(output.detach().cpu(),labels.numpy(),k)
                hr[k] += this_hr
                mrr[k] += this_mrr
            test_num += items.shape[0]

        for k in hr.keys():
            hr[k] /= test_num
            mrr[k] /= test_num

            print('Hit@%d %.3f'%(k, hr[k]*100))
            print('MRR@%d %.3f'%(k, mrr[k]*100))

            if hr_max[k] < hr[k]: hr_max[k] = hr[k]
            if mrr_max[k] < mrr[k]: 
                mrr_max[k] = mrr[k]
                e_stop = 0
            else:
                e_stop += 1 
        
        if e_stop >= args.patience:
            print("========Best Score===========")
            for k in hr.keys(): 
                print('Hit@%d %.3f'%(k, hr_max[k]*100))
                print('MRR@%d %.3f'%(k, mrr_max[k]*100)) 
            print('All time %d s'%(time.time()-start_time))
            break


