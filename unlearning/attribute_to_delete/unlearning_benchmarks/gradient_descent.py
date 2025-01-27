import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
#from ....evaluators import ThresholdMIA

from unlearning.unlearning_benchmarks import Unlearner
from unlearning.unlearning_benchmarks import benchmarks

from torch import optim
from torch import nn
import timeit
import copy

def eval_fool(model, intended_classes, target_loader, DEVICE):
    model.eval()
    for inputs, labels, index in target_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            output = model(inputs)
            predictions = torch.argmax(output.data, dim=1)
    if predictions[0] != intended_classes:
        # print("Target is not fooled ...")
        fooled = False
    else:
        # print("Target is fooled!")
        fooled = True
    return fooled * 1


class GD(Unlearner):
    """
    Provides model based on gradient descent unlearning.
    """

    def __init__(self,
                 model,
                 lr: float = 5e-5,
                 epochs: int = 10,
                 lr_scheduler=None,
                 momentum: float = 0.9,
                 weight_decay: float = 5e-4,
                 noise_var: float = 0.000,
                 device: str = "cpu",
                 **kwargs) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        #self.config = config
        self.lr = lr
        self.noise_var = noise_var
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs

        self.optimizer = optim.SGD(model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.DEVICE = device
        if self.lr_scheduler is None:
            print("No LR scheduling is used")
            self.lr_scheduler = lr_scheduler
        elif self.lr_scheduler == "Cosine-Annealing":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs)
        else:
            raise NotImplementedError(
                "This schedule has not been implemented, yet.")

        model_copy = copy.deepcopy(model)
        super(GD, self).__init__(model_copy)

    def _eval_model(self, model, loader):
        correct_test = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, labels, idx = data
                inputs = inputs.to(self.DEVICE)
                labels = labels.to(self.DEVICE)
                outputs = model(inputs)
                pred = outputs.max(1, keepdim=True)[1]
                correct_test += pred.eq(labels.view_as(pred)).sum().item()
            acc = correct_test / len(loader.dataset)
            print(f'ACCURACY - {acc}')
        return acc

    def get_updated_model(self,
                          retain_loader,
                          validation_loader=None,
                          mia_forget_loader=None,
                          mia_test_loader=None,
                          **kwargs):
        """
        Unlearning by fine-tuning.
        Args:
          net : nn.Module. pre-trained model to use as base of unlearning.
          retain_set : torch.utils.data.DataLoader.
            Dataset loader for access to the retain set. This is the subset
            of the training set that we don't want to forget.
          forget_set : torch.utils.data.DataLoader.
            Dataset loader for access to the forget set. This is the subset
            of the training set that we want to forget. This method doesn't
            make use of the forget set.
          validation_set : torch.utils.data.DataLoader.
            Dataset loader for access to the validation set. This method doesn't
            make use of the validation set.
        Returns:
          net : updated model
          wall_clock_time : time taken to update model
          accs : accuracies across train, test, forget points
            
        NOTE: 
            currently 
                "mia_forget_loader" and "mia_test_loader" are not called

        """
        print("Start training ...")
        model_device = next(self.model.parameters()).device

        # move retain set to model device

        accs = {'test': [], 'train': [], 'forget': []}
        tmias = {'scores': []}
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        # continue training using gradient descent on retain set
        start_time = timeit.default_timer()
        it = 0
        fooled_vec = {'epoch': [], 'step': []}
        print(f"training for {self.epochs}")
        for epoch in range(self.epochs):
            print(f"training - {epoch}")
            for inputs, targets, idx in retain_loader:
                # move to model device
                self.model.train()

                inputs, targets = inputs.to(model_device), targets.to(
                    model_device)
                #inputs, targets = inputs.to(self.DEVICE), targets.to( self.DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if self.noise_var > 0:
                    # get flatten param vector
                    param_vector = parameters_to_vector(
                        self.model.parameters())
                    # add noise to param vector
                    noise = torch.randn(
                        len(param_vector)).to(self.DEVICE) * torch.sqrt(
                            torch.tensor(self.noise_var)).to(self.DEVICE)
                    param_vector.add_(noise)
                    # load params back to model
                    vector_to_parameters(param_vector, self.model.parameters())
                self.optimizer.step()
                # add adv eval: this one is cheap to compute!
                it += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        wall_clock_time = timeit.default_timer() - start_time
        # for last step
        if validation_loader is not None:
            accs['test'].append(self._eval_model(self.model,
                                                 validation_loader))
        self.model.eval()
        # evaluate updated model
        # print('Evaluate model utility ...')
        # accs['train'] = self._eval_model(self.model, retain_set)
        # accs['test'] = self._eval_model(self.model, validation_set)
        # accs['forget'] = self._eval_model(self.model, forget_set)
        return self.model, wall_clock_time, accs
