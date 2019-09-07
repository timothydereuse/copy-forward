import datetime
import torch
import torch.nn as nn
import numpy as np
import parse_PPDD as pd
import pianoRollConvNet as prcn
import test_correlation as tc
import pickle
from importlib import reload
reload(pd)
reload(prcn)
reload(tc)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fname = 'models/run {}.txt'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
batch_size = 512

print('parsing data...')
ids, data = pd.parse_PPDD(limit=5000)


# given a prime, its continuation, and a translation t in that prime
# return the last window of the prime and the translated window from the start of the prime
# AND the naive-matching distance between the continuation and the predicted continuation at that spot.
def get_training_example(prime, cont, t, prime_bounds, window_size=cont_length_default):
    right_limit = prime_bounds[1]
    left_limit = prime_bounds[0]

    if t == 'best':
        t, _ = find_best_possible_translation(prime, cont, prime_bounds, window_size)

    # if the prime has too few notes in it just extend the left limit of it until it fits
    max_translate_amt = int(right_limit - left_limit - window_size * 2)
    if max_translate_amt < window_size:
        left_limit = right_limit - window_size * 3 # ensures at least @window_size number of possible translations
        max_translate_amt = window_size

    if t >= max_translate_amt:
        t = t % max_translate_amt

    base_roll = point_cloud_to_roll(prime, right_limit - window_size)
    window_slide_roll = point_cloud_to_roll(prime, left_limit + t)

    predict_l = left_limit + t + window_size
    prediction = extract_by_time_range(prime, predict_l, window_size)
    prediction[:, 0] = prediction[:, 0] + right_limit - predict_l
    accuracy = rolls_match(cont, prediction)['F1']

    stack_rolls = np.stack([base_roll, window_slide_roll], axis=2)
    return stack_rolls, accuracy


def find_best_possible_translation(prime, cont, bounds, window_size):
    right_limit = bounds[1]
    left_limit = bounds[0]
    max_translate_steps = int(right_limit - left_limit - 2 * window_size)
    translate_amts = np.arange(0, max_translate_steps, 1)

    best_translation = -1
    best_acc = 0

    for t in translate_amts:
        predict_l = left_limit + t + window_size
        prediction = extract_by_time_range(prime, predict_l, window_size)

        # translate to start of true continuation
        prediction[:, 0] = prediction[:, 0] + right_limit - predict_l
        accuracy = rolls_match(cont, prediction)['F1']

        if accuracy > best_acc:
            best_acc = accuracy
            best_translation = t

    return best_translation, best_acc


def point_cloud_to_roll(inp, start_time, lowest=25, height=80, length=80):

    out = np.zeros((length, height), dtype='bool')
    # add all notes whose times are between start_time and start_time + length
    for point in inp:
        time = point[0]
        note = point[1]
        if time < start_time or time >= (start_time + length):
            continue
        elif note < lowest:
            out[time - start_time][0] = True
        elif note >= lowest + height:
            out[time - start_time][height - 1] = True
        else:
            out[time - start_time][note - lowest] = True

    return out

def get_batch(ids, data, batch_size=batch_size):

    items = []
    scores = []

    chosen_ids = np.random.choice(ids, batch_size // 2)
    for id in chosen_ids:

        prime = data[id]['prime'][:, [0, 1, 4]]
        cont = data[id]['cont'][:, [0, 1, 4]]
        bounds = (min(prime[:, 0]), max(prime[:, 0]))
        channel_nums = set(prime[:, 2])

        for channel in channel_nums:
            t = np.random.randint(10000)
            rolls, acc = get_training_example(prime, cont, t, bounds)
            items.append(rolls)
            scores.append(acc)
            # good_rolls, acc = tc.get_training_example(prime, cont, 'best', bounds)
            # items.append(rolls)
            # scores.append(acc)

        if len(items) >= batch_size:
            break

    items = np.stack(items[:batch_size], axis=0)
    scores = np.stack(scores[:batch_size], axis=0)

    return items, scores

print('generating batch...')
train_data, train_labels = get_batch(ids, data, 256)

print('making model...')
model = prcn.pianoRollConvNet(img_size=train_data[0].shape)
model.to(device)

losses = []
train_data = train_data[:, None, :, :, :]
x_train = torch.tensor(train_data).float().to(device)
y_train = torch.tensor(train_labels).float().to(device)

learning_rate = 3e-4

loss_func = nn.MSELoss()
eval_loss_func = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute loss
    y_pred = y_pred[:, 0]
    loss = loss_func(y_pred, y_train)
    losses.append(loss.item())
    print(loss)

    # Reset gradients to zero, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()