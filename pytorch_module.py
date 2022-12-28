import torch
from torch.nn import Module
from torchweights import TorchWeights


class BaseTorchModule(Module):
    def __init__(self):
        super(BaseTorchModule, self).__init__()

    def set_weights(self, weights):
        input_dict = dict()
        for name, value in weights.items():
            input_dict["name"] = torch.tensor(value, requires_grad=True)
        self.load_state_dict(input_dict)

    def get_weights(self):
        weights = dict()
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        t_weights = TorchWeights(weights)
        return t_weights

    def set_weights_by_names(self, named_weights):
        input_dict_name = dict()
        for name, param in self.named_parameters():
            input_dict_name[name] = param.detach().cpu()
        for name, v in named_weights.items():
            input_dict_name[name] = torch.tensor(v, requires_grad=True)
        self.load_state_dict(input_dict_name)

    def get_weights_by_names(self, names):
        weights = dict()
        for name, param in self.named_parameters():
            if name in names:
                weights[name] = param.detach().cpu().numpy()
        t_weights = TorchWeights(weights)
        return t_weights

    def reset_parameters(self, mean=0.0, std=0.03):
        """
        默认的初始化，最好还是在子类中自己定义，因为不同的模型需要的初始化其实是不同的
        :param mean:
        :param std:
        :return:
        """
        for weight in self.parameters():
            weight.data.normal_(mean, std)
