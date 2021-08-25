import PreActResNet18
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


batch_size = 1
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

test_data = datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

net = PreActResNet18.PreActResNet18()
epsilons = [0.01, 0.02, 0.031, 0.04, 0.05, 0.06]
net = PreActResNet18.PreActResNet18()
check_point = torch.load("CIFAR10_PreActResNet18.checkpoint")
device = torch.device("cuda")
model = net.to(device)
model.load_state_dict(check_point['state_dict'])
model.eval()
criterion = nn.CrossEntropyLoss()


def pgd_attack(model, test_loader, epsilon=0.031, alpha=2/255, iterations=10):
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue
        orig_data = data
        pred = 0
        for t in range(iterations):
            data.requires_grad = True
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            if pred.item() != target.item():
                break
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            sign_data_grad = data_grad.sign()
            adv_data = data + alpha * sign_data_grad
            eta = torch.clamp(adv_data - orig_data, -epsilon, epsilon)
            data = torch.clamp(orig_data + eta, 0, 1).detach()
        if pred.item() == target.item():
            correct += 1
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc


step_size = 8/255
inner_iteration = 10
for eps in epsilons:
    pgd_attack(model, test_loader, eps, step_size, inner_iteration)
