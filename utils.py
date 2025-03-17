import torch 
import os 

def save_checkpoint(model,directory):
    dir=os.path.dirname(directory)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(),directory)

def resume_checkpoint(model,directory,device_id):
    state_dict=torch.load(directory,map_location=lambda storage,loc: storage.cuda(device_id))
    model.load_state_dict(state_dict)

def use_gpu(enabled,device_id=None):
    if enabled:
        if torch.backends.mps.is_available():
            device=torch.device("mps")

        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                    lr=params['adam_lr'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


