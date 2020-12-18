import torch
import torch.nn as nn



class L2Regularization(nn.Module):
    def __init__(self, model, weight_decay):
        
        super(L2Regularization, self).__init__()
        assert weight_decay > 0
        self.model = model
        self.weight_decay = weight_decay
        # self.p = p
        self.weight_list= self.get_weight(model)
        self.weight_info(self.weight_list)
    
    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay)
        return reg_loss
    
    def regularization_loss(self, weight_list, weight_decay):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=2)
            reg_loss = reg_loss + l2_reg
        
        return weight_decay * reg_loss
    
    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    
    def weight_info(self, weight_list):
        print("-----regularization weight-----")
        for name, w in weight_list:
            print(name)
        print("-------------------------------")