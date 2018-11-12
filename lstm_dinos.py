'''Example script to generate text from Nietzsche's writings.

Multi to one char
 Or better fixed to one char with a fixed input window

5
[(0.5, 23), (1.0, 40), (1.2, 43), (0.2, 21)]

3
[(0.5, 49), (1.0, 59), (1.2, 59), (0.2, 43)]

2
[(0.5, 57), (1.0, 60), (1.2, 60), (0.2, 52)]

8
[(0.5, 20), (1.0, 33), (1.2, 40), (0.2, 14)]



'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

# path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# path = get_file('pride.txt', origin='http://www.gutenberg.org/files/1342/1342-0.txt')
path = get_file('dinos.txt', origin='https://raw.githubusercontent.com/appchamp/ml/master/dinos.txt')

with io.open(path, encoding='utf-8') as f:
    text = f.read()
print('corpus length:', len(text))

chars = sorted(list(set(text+' ')))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
# print(char_indices.items())

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 8
step = 1
sentences = []
next_chars = []
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])
# print('nb sequences:', len(sentences))

dino_dict = {}
for dino in text.split('\n'):
    dino_dict[dino] = 1
    newdino = ' ' * maxlen + dino + "\n"
    for i in range(0, len(dino) +1):
        sentences.append(newdino[i: i+maxlen])
        next_chars.append(newdino[i+maxlen])

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    #start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in diversity_list:
        print('----- diversity:', diversity)

        generated = ''
        sentence = ' ' * maxlen
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            if (next_char == '\n'):
                break

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

        if dino_dict.has_key(generated):
            print("From the Dict")
        else:
            print("New")
            diversity_map[diversity] += 1

    print(diversity_map.items())

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

diversity_map = {}
diversity_list = [0.2, 0.5, 1.0, 1.2]
for d in diversity_list:
    diversity_map[d] = 0

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
