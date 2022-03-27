# -*- coding: utf-8 -*-

def print_model_summary(model):
    """
    打印每一个层的参数信息
    :return:
    """
    # for name, parameter in self.named_parameters():
    #     print("name: {}, parameter shape: {}".format(name, parameter.numel()))
    if not model:
        raise Exception("参数model不能为空!")
    module_parameter_dict = dict((k, v.numel()) for k, v in model.named_parameters())
    print("module_parameter_dict: {}".format(module_parameter_dict))
    print_header = "name" + (" " * 8) + "module" + (" " * 8) + "#parameters"
    print(print_header)
    print("=" * len(print_header) * 4)

    for name, module in model.named_modules():
        parameter_dict = dict(
            filter(lambda item: str(item[0]).__contains__(name), module_parameter_dict.items())) if len(
            name) > 0 else {}
        name = name if len(name) > 0 else "None"
        print("{}        {}        {}".format(name, module, parameter_dict))
        print("_" * len(print_header) * 4)
    print("=" * len(print_header) * 4)
    print("Total params: {}".format(sum(module_parameter_dict.values())))