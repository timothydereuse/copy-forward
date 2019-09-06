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
            rolls, acc = tc.get_training_example(prime, cont, t, bounds)
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