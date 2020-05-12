import gc
import torch
def monitor_memory():
    tensor_list = []
    tensor_graph_number = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
                if obj.requires_grad:
                    tensor_graph_number += 1
        except:
            pass
    print("memory state")
    print(f'total tensors number = {len(tensor_list)}.')
    print(f'graph tensors number = {tensor_graph_number}.')
    if torch.cuda.is_available():
        print(len(torch.cuda.memory_stats()))
        print(torch.cuda.memory_reserved() / 1024 ** 2, torch.cuda.max_memory_reserved() / 1024 ** 2)
        print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.max_memory_allocated() / 1024 ** 2)
    return len(tensor_list), tensor_graph_number, tensor_list