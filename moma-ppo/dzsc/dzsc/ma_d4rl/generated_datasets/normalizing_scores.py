REACHER_EXPERT_SCORE = -4.2370844
REACHER_RANDOM_SCORE = -11.144625



def normalize_scores(scores, task_name):
    if 'reacher' in task_name:
        random_score = REACHER_RANDOM_SCORE
        expert_score = REACHER_EXPERT_SCORE
    else:
        raise NotImplementedError

    return (scores - random_score) / (expert_score - random_score)
