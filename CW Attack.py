import PreActResNet18
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim


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
net = PreActResNet18.PreActResNet18()
check_point = torch.load("CIFAR10_PreActResNet18.checkpoint")
device = torch.device("cuda")
model = net.to(device)
model.load_state_dict(check_point['state_dict'])
model.eval()
criterion = nn.CrossEntropyLoss()


def cw_attack(model, image, label, targeted=False, c=1e-4, kappa=0, max_iter=1000,
              learning_rate=0.01):

    def f(x):
        output = model(x)
        one_hot_label = torch.eye(len(output[0]))[label].to(device)
        i, _ = torch.max((1 - one_hot_label) * output, dim=1)
        j = torch.masked_select(output, one_hot_label.bool())
        if targeted:
            return torch.clamp(i - j, min=-kappa)
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(image, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10
    for step in range(max_iter):
        a = 1/2 * (nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, image)
        loss2 = torch.sum(c * f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if step % (max_iter // 10) == 0:
            if cost > prev:
                return a
            prev = cost
        print('- Learning progress: %2.2f %%' % ((step + 1) / max_iter * 100), end='\r')
    attack_image = 1/2 * (nn.Tanh()(w) + 1)
    return attack_image


correct = 0
total = 0
for data, target in test_loader:
    total += 1
    data, target = data.to(device), target.to(device)
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    if init_pred.item() != target.item():
        continue
    data = cw_attack(model, data, target, targeted=False, c=0.1)
    output = model(data)
    _, pred = torch.max(output.data, 1)
    correct += (pred == target).sum()
    if total % 100 == 0:
        final_acc = correct / float(total)
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))
