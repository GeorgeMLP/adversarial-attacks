import PreActResNet18
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets


def coordinate_newton(losses, indice, grad, hess, batch_size, mt_arr, vt_arr,
                      real_modefier, adam_epoch, up, down, step_size, beta1,
                      beta2, proj):
    cur_loss = losses[0]
    for i in range(batch_size):
        grad[i] = (losses[i * 2 + 1] - losses[i * 2 + 2]) / 0.0002
        hess[i] = (losses[i * 2 + 1] - 2 * cur_loss + losses[i * 2 + 2]) / (0.0001 * 0.0001)
    hess[hess < 0] = 1.0
    hess[hess < 0.1] = 0.1
    m = real_modefier.reshape(-1)
    old_val = m[indice]
    old_val -= step_size * grad / hess
    if proj:
        old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
    m[indice] = old_val


def loss_run(input, target, model, modifier, use_tanh, use_log, targeted,
             confidence, const):
    if use_tanh:
        pert_out = torch.tanh(input + modifier) / 2
    else:
        pert_out = input + modifier
    output = model(pert_out)
    if use_log:
        output = F.softmax(output, -1)
    if use_tanh:
        loss1 = torch.sum(torch.square(pert_out - torch.tanh(input) / 2), dim=(1, 2, 3))
    else:
        loss1 = torch.sum(torch.square(pert_out - input), dim=(1, 2, 3))
    real = torch.sum(target * output, -1)
    other = torch.max((1 - target) * output - (target * 10000), -1)[0]
    if use_log:
        real = torch.log(real + 1e-30)
        other = torch.log(other + 1e-30)
    confidence = torch.tensor(confidence).type(torch.float64).cuda()
    if targeted:
        loss2 = torch.max(other - real, confidence)
    else:
        loss2 = torch.max(real - other, confidence)
    loss2 = const * loss2
    l2 = loss1
    loss = loss1 + loss2
    return loss.detach().cpu().numpy(), l2.detach().cpu().numpy(), loss2.detach().cpu().numpy(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()


def l2_attack(input, target, model, targeted, use_log, use_tanh, solver,
              reset_adam_after_found=True, abort_early=True, batch_size=128,
              max_iter=1000, const=0.01, confidence=0.0, early_stop_iters=100,
              binary_search_steps=9, step_size=0.01, adam_beta1=0.9,
              adam_beta2=0.999):
    early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10
    input = torch.from_numpy(input).cuda()
    target = torch.from_numpy(target).cuda()
    var_len = input.view(-1).size()[0]
    modifier_up = np.zeros(var_len, dtype=np.float32)
    modifier_down = np.zeros(var_len, dtype=np.float32)
    real_modifier = torch.zeros(input.size(), dtype=torch.float32).cuda()
    mt = np.zeros(var_len, dtype=np.float32)
    vt = np.zeros(var_len, dtype=np.float32)
    adam_epoch = np.ones(var_len, dtype=np.int32)
    grad = np.zeros(batch_size, dtype=np.float32)
    hess = np.zeros(batch_size, dtype=np.float32)
    upper_bound = 1e10
    lower_bound = 0.0
    out_best_attack = input.clone().detach().cpu().numpy()
    out_best_const = const
    out_bestl2 = 1e10
    out_bestscore = -1
    if use_tanh:
        input = torch.atanh(input * 0.99999)
    if not use_tanh:
        modifier_up = -input.clone().detach().view(-1).cpu().numpy()
        modifier_down = -1 - input.clone().detach().view(-1).cpu().numpy()

    def compare(x, y):
        if not isinstance(x, (float, int, np.int64)):
            if targeted:
                x[y] -= confidence
            else:
                x[y] += confidence
            x = np.argmax(x)
        if targeted:
            return x == y
        else:
            return x != y

    for step in range(binary_search_steps):
        bestl2 = 1e10
        prev = 1e6
        bestscore = -1
        last_loss2 = 1.0
        mt.fill(0)
        vt.fill(0)
        adam_epoch.fill(1)
        stage = 0
        for iter in range(max_iter):
            if (iter + 1) % 100 == 0:
                loss, l2, loss2, _, __ = loss_run(input, target, model, real_modifier,
                                                  use_tanh, use_log, targeted, confidence,
                                                  const)
            var_list = np.array(range(0, var_len), dtype=np.int32)
            indice = var_list[np.random.choice(var_list.size, batch_size,
                                               replace=False)]
            var = np.repeat(real_modifier.detach().cpu().numpy(), batch_size * 2 + 1,
                            axis=0)
            for i in range(batch_size):
                var[i * 2 + 1].reshape(-1)[indice[i]] += 0.0001
                var[i * 2 + 2].reshape(-1)[indice[i]] -= 0.0001
            var = torch.from_numpy(var)
            var = var.view((-1, ) + input.size()[1:]).cuda()
            losses, l2s, losses2, scores, pert_images = loss_run(input, target, model, var,
                                                                 use_tanh, use_log, targeted,
                                                                 confidence, const)
            real_modifier_numpy = real_modifier.clone().detach().cpu().numpy()
            if solver == "newton":
                coordinate_newton(losses, indice, grad, hess, batch_size, mt, vt,
                                  real_modifier_numpy, adam_epoch, modifier_up,
                                  modifier_down, step_size, adam_beta1, adam_beta2,
                                  proj=not use_tanh)
            real_modifier = torch.from_numpy(real_modifier_numpy).cuda()
            if losses2[0] == 0.0 and last_loss2 != 0.0 and stage == 0:
                if reset_adam_after_found:
                    mt.fill(0)
                    vt.fill(0)
                    adam_epoch.fill(1)
                stage = 1
            last_loss2 = losses2[0]
            if abort_early and (iter + 1) % early_stop_iters == 0:
                if losses[0] > prev * .9999:
                    break
                prev = losses[0]
            if l2s[0] < bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(), -1)):
                bestl2 = l2s[0]
                bestscore = np.argmax(scores[0])
            if l2s[0] < out_bestl2 and compare(scores[0], np.argmax(target.cpu().numpy(), -1)):
                out_bestl2 = l2s[0]
                out_bestscore = np.argmax(scores[0])
                out_best_attack = pert_images[0]
                out_best_const = const
                return out_best_attack, out_bestscore
        if compare(bestscore, np.argmax(target.cpu().numpy(), -1)) and bestscore != -1:
            upper_bound = min(upper_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
        else:
            lower_bound = max(lower_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10
    return out_best_attack, out_bestscore


def generate_data(test_loader, targeted, samples, start):
    inputs = []
    targets = []
    num_label = 10
    cnt = 0
    for i, data in enumerate(test_loader):
        if cnt < samples:
            if i > start:
                data, label = data[0], data[1]
                if targeted:
                    seq = range(num_label)
                    for j in seq:
                        if j == label.item():
                            continue
                        inputs.append(data[0].numpy())
                        targets.append(np.eye(num_label)[j])
                else:
                    inputs.append(data[0].numpy())
                    targets.append(np.eye(num_label)[label.item()])
                cnt += 1
            else:
                continue
        else:
            break
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets


def attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device):
    r = []
    for i in range(len(inputs)):
        attack, score = l2_attack(np.expand_dims(inputs[i], 0), np.expand_dims(targets[i], 0),
                                  model, targeted, use_log, use_tanh, solver, device)
        r.append(attack)
    return np.array(r)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    device = torch.device("cuda")
    model = PreActResNet18.PreActResNet18().to(device)
    check_point = torch.load("CIFAR10_PreActResNet18.checkpoint")
    model.load_state_dict(check_point['state_dict'])
    model.eval()
    use_log = True
    use_tanh = True
    targeted = False
    solver = "newton"
    cnt = 0
    total = 0
    for i in range(10000):
        total += 1
        inputs, targets = generate_data(test_loader, targeted, samples=1, start=i)
        out = model(torch.from_numpy(inputs).cuda())
        if np.argmax(out.detach().cpu().numpy(), -1) != np.argmax(targets, -1):
            continue
        adv = attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device)
        adv_out = model(torch.from_numpy(adv).cuda())
        target_class = np.argmax(targets, -1)
        adv_class = np.argmax(adv_out.detach().cpu().numpy(), -1)
        cnt += (adv_class == target_class).sum()
        if total % 100 == 0:
          print("Test Accuracy:", cnt / total)
    print("Success Rate:", (1 - cnt / total) * 100.0)
