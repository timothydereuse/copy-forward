import numpy as np
from collections import Counter


def evaluate_tec(original, generated):
    '''
    given a list of original and generated events, calculate
    precision and recall based on translation equivalence.
    '''
    translation_vectors = []
    generated_vec = np.array([(float(s[0]), int(s[1])) for s in generated])
    original_list = [(float(s[0]), int(s[1])) for s in original]
    for i in original_list:
        vectors = generated_vec - i
        translation_vectors.extend([tuple(v) for v in vectors])
    grouped_vectors = dict(Counter(translation_vectors))
    max_translation = max([grouped_vectors[k] for k in grouped_vectors])

    recall = (max_translation - 1) / float(len(original) - 1)
    precision = (max_translation - 1) / float(len(generated) - 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * recall * precision) / (
            recall + precision
        )
    output = {'rec': recall, 'prec': precision, 'F1': f1}
    return output


def evaluate_continuation(
    original,
    generated,
    last_onset_prime,
    onset_increment,
    evaluate_from_onset,
    evaluate_until_onset):
    """ Given the original and the generated continuation
    of the test item (in onset/pitch dictionaries),
    collect the following events and get tec score.
    'onset_increment' determines the increase of onset steps in evaluation.
    For the first ioi, the last event of the prime is also required.
    'evaluate_until_onset' determines until how many quarter notes
    after the cut-off point we evaluate.
    """
    scores = {'precision': {}, 'recall': {}, 'F1': {}}
    no_steps = int((evaluate_until_onset - evaluate_from_onset) / onset_increment)
    max_onset = evaluate_until_onset + last_onset_prime
    for step in range(no_steps+1):
        onset = step * onset_increment + evaluate_from_onset
        cutoff = last_onset_prime + onset
        if cutoff <= max_onset:
            original_events = [o for o in original if float(o[0]) <= cutoff]
            generated_events = [
                g for g in generated if float(g[0]) <= cutoff
            ]
            if (len(original_events)<=1 or len(generated_events)<=1):
                scores['precision'][onset] = None
                scores['recall'][onset] = None
                scores['F1'][onset] = None
            else:
                output = evaluate_tec(original_events, generated_events)
                scores['precision'][onset] = output['prec']
                scores['recall'][onset] = output['rec']
                scores['F1'][onset] = output['F1']
    return scores
