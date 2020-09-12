import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch


def one_pixel_attack(model, imgs, true_labels, class_names, target_label=None, iters=100, pop_size=400, verbose=False):
    """
    *** NOT BATCHED ATM
    Runs one-pixel attack on a model.
    One pixel attack uses differential evolution to fool a network into misclassifying the input image.
    The atack can be done in two modes:
    Targetted: make the image be recognized as that class (early stop > 50%)
    Untargetted: make the image be recognized as any OTHER class (early stop < 5%)
    From:
    "One Pixel Attack for Fooling Deep Neural Networks"
    https://arxiv.org/pdf/1710.08864.pdf
    Original code this is based on:
    https://github.com/nitarshan/one-pixel-attack/blob/master/One%20Pixel%20Attack%20for%20Fooling%20Deep%20Neural%20Networks.ipynb
    :param model: Should output (b, nb_classes) where values are logits
    :param img:
    :param true_label:
    :param class_names:
    :param target_label:
    :param iters:
    :param pop_size:
    :param verbose:
    :return:
    """
    # TODO: make batched
    img = imgs[0]
    true_label = true_labels[0]

    # bookkeeping
    is_targeted = target_label is not None
    label = target_label if is_targeted else true_label

    # pick some pixels to start with
    candidates = np.random.random((pop_size, 5))
    candidates[:, 2:5] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 3)), 0, 1)

    # see how the network evals this image
    if verbose:
        print('running baseline fitness')

    fitness = _evaluate(candidates, img, label, model)

    def is_success():
        return (is_targeted and fitness.max() > 0.5) or ((not is_targeted) and fitness.min() < 0.05)

    if verbose:
        print('running evolutionary attack...')

    for iteration in range(iters):
        # Early Stopping
        if is_success():
            break

        # Print progress
        if verbose == True and iteration %10 == 0:
            print("Target Probability [Iteration {}]:".format(iteration), fitness.max() if is_targeted else fitness.min())

        # Generate new candidate solutions
        new_gen_candidates = _evolve(candidates, strategy="resample")

        # Evaluate new solutions
        new_gen_fitness = _evaluate(new_gen_candidates, img, label, model)

        # Replace old solutions with new ones where they are better
        successors = new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
        candidates[successors] = new_gen_candidates[successors]
        fitness[successors] = new_gen_fitness[successors]

    # track the best fooling pixel
    best_idx = fitness.argmax() if is_targeted else fitness.argmin()
    best_solution = candidates[best_idx]
    best_score = fitness[best_idx]

    if verbose == True:
        _visualize_perturbation(best_solution, img, true_label, model, class_names, target_label)

    result = _generate_sample_perturbation(best_solution, img, true_label, model, class_names, target_label)
    return is_success(), best_solution, best_score, result


def _evaluate(candidates, img, label, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for i, xs in enumerate(candidates):
            p_img = _perturb(xs, img)

            batch = [p_img.unsqueeze(0), label.unsqueeze(0)]
            y_hat = model(batch).squeeze(0)
            y_hat = y_hat[label].item()

            preds.append(y_hat)
    return np.array(preds)


def _evolve(candidates, F=0.5, strategy="clip"):
    gen2 = candidates.copy()
    num_candidates = len(candidates)
    for i in range(num_candidates):
        x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
        x_next = (x1 + F*(x2 - x3))
        if strategy == "clip":
            gen2[i] = np.clip(x_next, 0, 1)
        elif strategy == "resample":
            x_oob = np.logical_or((x_next < 0), (1 < x_next))
            x_next[x_oob] = np.random.random(5)[x_oob]
            gen2[i] = x_next
    return gen2


def _perturb(p, img):
    # p should be [0, 1]
    assert np.alltrue(p >= 0.0)
    assert np.alltrue(p <= 1.0)

    # image should be c, h, w
    c, h, w = img.size()

    # copy result img
    p_img = img.clone()

    # pick perturbation
    xy = (p[0:2].copy() * h).astype(int)
    xy = np.clip(xy, 0, h-1)
    rgb = p[2:5].copy()
    rgb = np.clip(rgb, 0, 1)

    # apply perturbation
    p_img[:,xy[0],xy[1]] = torch.from_numpy(rgb)

    return p_img

def _generate_sample_perturbation(p, img, label, model, class_names, target_label=None):
    p_img = _perturb(p, img)
    summary_stats = _summarize_pred(p_img, label, model, class_names, target_label)

    result = {
        'perturbed_img': p_img,
    }
    result.update(summary_stats)
    return result

def _visualize_perturbation(p, img, label, model, class_names, target_label=None):
    p_img = _perturb(p, img)
    print("Perturbation:", p)
    _show(p_img)
    _summarize_pred(p_img, label, model, class_names, target_label)


def _show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()


def _summarize_pred(img, label_i, model, class_names, target_label=None):
    batch = [img.unsqueeze(0), label_i.unsqueeze(0)]
    y_hat = model(batch).squeeze(0)

    prediction = y_hat.max(-1)[1]
    prediction_class_i = prediction[0].item()
    prediction_name = class_names[prediction_class_i]
    label_probs = F.softmax(y_hat.squeeze(), dim=0)
    true_label_p = label_probs[label_i].item()

    result = {
        'true_label': f'{class_names[label_i]}_{label_i}',
        'prediction': f'{prediction_name}_{prediction_class_i}',
        'label_probs': label_probs,
        'true_label_p': true_label_p

    }

    if target_label is not None:
        target_label_p = label_probs[target_label].item()
        result['target_label_p'] = target_label_p

    return result


class TestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32*32*3, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = TestModel()
    imgs = torch.FloatTensor(5, 3, 32, 32).uniform_(0, 1)
    labels = torch.tensor([0,1,0,2,0])
    class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    one_pixel_attack(
        model=model,
        imgs=imgs,
        true_labels=labels,
        class_names=class_names,
        verbose=True
    )