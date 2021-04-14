import torch

__all__ = ["index_to_mask", "random_split_mask_per_class"]

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_split_mask_per_class(num_instance,
                                labels,
                                num_per_class_train=30,
                                num_per_class_val=30):
    device = labels.device
    num_labels = int(labels.max().item() + 1)
    indices = []
    for i in range(num_labels):
        index = torch.nonzero(labels == i).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index.to(device))
    
    bd1, bd2 = num_per_class_train, num_per_class_train + num_per_class_val
    train_index = torch.cat([i[:bd1] for i in indices], dim=0)
    val_index = torch.cat([i[bd1:bd2] for i in indices], dim=0)
    test_index = torch.cat([i[bd2:] for i in indices], dim=0)

    train_mask = index_to_mask(train_index, size=num_instance)
    val_mask = index_to_mask(val_index, size=num_instance)
    test_mask = index_to_mask(test_index, size=num_instance)
    return train_mask, val_mask, test_mask
