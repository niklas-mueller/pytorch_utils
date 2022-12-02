import torch
import torchvision
import torch.nn as nn
import numpy as np
import time
from matplotlib.backends.backend_pdf import PdfPages # This is stupid but needs to be done in order to save as pdf pages
from matplotlib import pyplot as plt
from albumentations.augmentations.transforms import ImageCompression

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    fig = plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    return fig

def visualize_layers(model):
    figs = []
    _model = model.module if type(model) == nn.DataParallel else model
    for name, module in _model.named_modules():
        if not isinstance(module, nn.Sequential):
            if type(module) == nn.modules.conv.Conv2d or type(module) == nn.Conv2d:
                filter = module.weight.cpu().data.clone()
            else:
                continue
            fig = visTensor(filter, ch=0, allkernels=True)
            figs.append(fig)
            plt.axis('off')
            plt.title(f'Layer: {name}')
            plt.ioff()

    return figs

def _get_device(model):
    if hasattr(model, 'src_device_obj'):
        device = model.src_device_obj
    elif hasattr(model, 'device'):
        device = model.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def rgb_to_opponent_space(img, normalize=False):
    """rgb_to_opponent_space

    Convert a image in RBG space to color opponent space, i.e., Intensity, Blue-Yellow (BY) opponent, Red-Green (RG) opponent.

    Parameters
    ----------
    img: ndarray, list, PIL Image
        Image to be converted
    normalize: bool, optional
        Whether to normalize pixel values by the maximum, by default False

    Returns
    ----------
    ndarray
        Array of length 3, with image converted to Intensity, BY, RG opponent, respectively.

    Example
    ----------
    >>> 
    """
    o1 = 0.3 * img[:, :, 0] + 0.58 * img[:, :, 1] + \
        0.11 * img[:, :, 2]   # Intensity/Luminance
    o2 = 0.25 * img[:, :, 0] + 0.25 * img[:, :, 1] - \
        0.5 * img[:, :, 2]   # BY opponent
    o3 = 0.5 * img[:, :, 0] - 0.5 * \
        img[:, :, 1]                        # RG opponent

    if normalize:
        ret = []
        for _x in [o1, o2, o3]:
            _max = _x.max()
            ret.append(_x / _max)
        return np.array(ret)

    return np.array([o1, o2, o3])

class ToJpeg(object):

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        jpeg = ImageCompression().apply(np.array(sample))

        return jpeg

class ToOpponentChannel(object):

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        ops = rgb_to_opponent_space(np.array(sample))

        return ops.transpose((1,2,0))


def get_result_figures(results:dict, model=None, result_manager=None, pdf_filename="result_figures.pdf", confusion_matrix=None):
    figs = []

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(results['training_losses'], label='Training')
    ax.plot(results['validation_losses'], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Losses during Training')
    figs.append(fig)

    ##############
    # Accuracy bar plots on 
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    train_acc = results['eval_trained_traindata']['accuracy']
    test_acc = results['eval_trained_testdata']['accuracy']

    ax.bar(x=[1,2], height=[train_acc,test_acc])
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Training', 'Testing'])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Dataset')
    figs.append(fig)
    ##############
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    train_losses = list(results['eval_trained_traindata']['batch_losses'].values())
    test_losses = list(results['eval_trained_testdata']['batch_losses'].values())

    ax.boxplot(x=[train_losses,test_losses])
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Training', 'Test'])
    ax.set_ylabel('Average Batch Loss')
    ax.set_xlabel('Dataset')
    figs.append(fig)
    ##############

    if confusion_matrix is not None:
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        bar = ax.imshow(confusion_matrix)
        plt.colorbar(bar, ax=ax, label='Number of occurences', fraction=0.046)
        ax.set_title(f"Confusion Matrix")
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Original Class')
        figs.append(fig)

    ##############

    if result_manager is not None:
        result_manager.save_pdf(figs=figs, filename=pdf_filename)
        

def get_confusion_matrix(model, loader, n_classes, device=None):
    if device is None:
        device = _get_device(model=model)
    confusion_matrix = np.zeros((n_classes, n_classes))# [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for img, label in loader:
        pred_labels = model(img.to(device)).max(1)[1]
        for index, _label in enumerate(label):
            _pred = int(pred_labels[index].detach())
            confusion_matrix[int(_label.detach())][_pred] += 1

    return confusion_matrix

def evaluate(loader, model:nn.DataParallel, criterion, device=None, verbose:bool=False):
    num_correct = 0
    num_samples = 0
    losses = {}
    model.eval()

    if device is None:
        device = _get_device(model=model)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            loss = criterion(scores, y)
            losses[i] = loss.cpu().tolist()
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        accuracy = float(num_correct)/float(num_samples)
        if verbose:
            print(f'Got {num_correct} \t/ {num_samples} correct -> accuracy {accuracy*100:.2f} %') 
    
    model.train()

    return {'accuracy': accuracy, 'batch_losses': losses}


def train(model, trainloader, valloader, loss_fn, optimizer, device, 
            n_epochs:int, lr_scheduler:torch.optim.lr_scheduler=None, plateau_lr_scheduler:torch.optim.lr_scheduler=None, result_manager=None, verbose:bool=True,
            testloader=None, visualize_layers:bool=False):
    

    current_time = time.ctime().replace(' ', '_')
    results = {}
    model.train(True)
    loss = None
    losses = []
    validation_losses = []
    epoch_times = []

    best_valid_loss = np.inf
    best_training_loss = np.inf
    best_model_state_dict = model.state_dict()
    for epoch in range(n_epochs):
        if verbose:
            print(f"Running epoch {epoch}")

        start_time_epoch = time.time()
        # Initalize losses
        # running_loss = 0.0
        # last_loss = 0.0

        # Training loop
        for inputs, labels in trainloader:

            # Move data to device
            inputs = inputs.to(device) #, dtype=torch.float32
            labels = labels.to(device) #, dtype=torch.float32

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()


        # Get time needed for epoch
        end_time_epoch = time.time()
        epoch_times.append(end_time_epoch - start_time_epoch)
        
        losses.append(loss.item())
        
        # Check if overall training loss has improve and if then save the model
        if loss.item() < best_training_loss:
            best_training_loss = loss.item()
            if verbose:
                print(f"Found new best model: Saving model in epoch {epoch} with loss {loss.item()}.")

            # Save model
            if result_manager is not None:
                result_manager.save_model(model, filename=f'best_model_{current_time}.pth', overwrite=True)

            best_model_state_dict = model.state_dict()
        if verbose:
            print(f"Loss after epoch {epoch}: {loss.item()}")

        # Evaluate on validation set
        eval = evaluate(loader=valloader, model=model, criterion=loss_fn, device=device, verbose=False)

        # Compute average over batches
        validation_loss = np.mean([batch_losses for _, batch_losses in eval['batch_losses'].items()])

        validation_losses.append(validation_loss)
        results[f'validation_during_training_epoch-{epoch}'] = eval

        # if plateau learning rate scheduler is given, make step depending on average validation loss
        if plateau_lr_scheduler is not None:
            plateau_lr_scheduler.step(validation_loss)

        # Check if validation loss has improved and if then store evaluation results
        if validation_loss < best_valid_loss:
            best_valid_loss = validation_loss
            

        # If learning rate scheduler is given, make step
        if lr_scheduler is not None:
            lr_scheduler.step()

    model.train(False)

    # Save final model (NOT necessarily best model)
    if result_manager is not None:
        result_manager.save_model(model, filename=f'final_model_{current_time}.pth')


    # Load best model
    model.load_state_dict(best_model_state_dict)
    
    
    # Get training losses and times
    results['training_losses'] = losses
    results['validation_losses'] = validation_losses
    results['epoch_times'] = epoch_times

    if verbose:
        print("\n\n-----------------------------\n\n")
        if loss is not None:
            print(f'Finished Training with loss: {loss.item()}')
        print(f'Average time per epoch for {n_epochs} epochs: {np.mean(epoch_times)}')

        # Evaluate on training model to get first indication whether training works
        training_eval = evaluate(loader=trainloader, model=model, criterion=loss_fn, verbose=True)
        results['eval_trained_traindata'] = training_eval

        print(f"Evaluated TRAINING data: Accuracy: {training_eval['accuracy']}")


    # Evaluate on test dataset
    if testloader is not None:
        eval = evaluate(loader=testloader, model=model, criterion=loss_fn, verbose=True)
        results['eval_trained_testdata'] = eval

        if verbose:
            print(f"Evaluated test data: Accuracy: {eval['accuracy']}")

    # Save all results that have been accumulated
    if result_manager is not None:
        result_manager.save_result(results, filename=f'training_results_{current_time}.yml')

        if verbose:
            print(f"Saved results and model state.")

    # Visualize the layers of the model
    if visualize_layers:
        figs = visualize_layers(model)
        if result_manager is not None:
            result_manager.save_pdf(figs=figs, filename='layer_visualisation_after_training.pdf')
        else:
            print("To save the visualisation provide a result manager!")

        if verbose:
            print(f"Saved visualization of layers after training.")

    if verbose:
        print(f"Done!")