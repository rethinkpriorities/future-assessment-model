import squigglepy as sq
import numpy as np

from pprint import pprint

from utils import compute


def get_anchor_gpt_benchmark_pct(anchor, target):
    out = round(np.mean([a <= target for a in anchor]) * 100, 1)
    return '<0.1%' if out == 0 else str(out) + '%'


def compare_anchor_to_gpt(anchor):
    print('-')
    print('GPT2 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-2'])))
    print('GPT3 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-3'])))
    print('GPT4 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-4'])))
    print('10x GPT4 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-4'] + 1)))
    print('100x GPT4 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-4'] + 2)))
    print('1000x GPT4 can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, compute['GPT-4'] + 3)))


def tai_log_flop_needs(brain, efficiency=0, transformative_vs_human=0, horizon_length=0, scaling_exponent=0,
                       flops_per_param_per_sec=0, params=None, ref_params=11.2, ref_params_samples=12,
                       bayes_update=None):
    params_ = brain + efficiency - flops_per_param_per_sec if params is None else params
    dist = ((brain + efficiency + transformative_vs_human + horizon_length + ref_params_samples) -
            (scaling_exponent * ref_params) + (scaling_exponent * params_))
    
    if bayes_update is None:
        return dist
    else:
        return sq.dist_fn(dist, bayes_update)


@np.vectorize
def cotra_bayes_update_against_low_flop(f):
    import random
    f = f + ~sq.norm(1,3) if f < 27 and random.random() > 0.3 else f
    f = f + ~sq.norm(1,3) if f < 26 and random.random() > 0.2 else f
    f = f + ~sq.norm(1,3) if f < 25 and random.random() > 0.1 else f
    f = f + ~sq.norm(1,3) if f < 24 else f
    f = 24 if f < 24 else f
    return f


@np.vectorize
def peter_bayes_update_against_low_flop(f):
    f = f + ~sq.lognorm(1,3) if f < 26 else f
    f = f + ~sq.lognorm(2,4) if f < 25 else f
    f = f + ~sq.lognorm(3,5) if f < 24 else f
    f = 24 if f < 24 else f
    return f


def cotra_anchor(horizon_length, bayes_update=cotra_bayes_update_against_low_flop, chinchilla=False):
    return tai_log_flop_needs(brain=sq.lognorm(11,19.5),
                              efficiency=1,
                              transformative_vs_human=sq.norm(-2,2),
                              horizon_length=horizon_length,
                              scaling_exponent=sq.norm(0.5, 1.5 if chinchilla else 1.1),
                              flops_per_param_per_sec=sq.norm(1,2),
                              bayes_update=bayes_update)


def plot_anchors(anchor1=None, anchor2=None, anchor3=None, anchor4=None, anchor5=None, anchor6=None,
                 label1=None, label2=None, label3=None, label4=None, label5=None, label6=None,
                 bins=100, alpha=0.6, verbose=True, xlim=[20, 75], figsize=(10,8), disable_ci_lines=False):
    if label1 is None:
        label1 = 'Anchor1'
    if label2 is None:
        label2 = 'Anchor2'
    if label3 is None:
        label3 = 'Anchor3'
    if label4 is None:
        label4 = 'Anchor4'
    if label5 is None:
        label5 = 'Anchor5'
    if label6 is None:
        label6 = 'Anchor6'
        
    if anchor1 is None:
        raise ValueError('Must define {}'.format(label1))
        
    if anchor2 is None and anchor3 is not None:
        raise ValueError('{} defined without defining {}'.foramt(label3, label2))

    if anchor3 is None and anchor4 is not None:
        raise ValueError('{} defined without defining {}'.foramt(label4, label3))

    if anchor4 is None and anchor5 is not None:
        raise ValueError('{} defined without defining {}'.foramt(label5, label4))

    if anchor5 is None and anchor6 is not None:
        raise ValueError('{} defined without defining {}'.foramt(label6, label5))

    anchors = [anchor1, anchor2, anchor3, anchor4, anchor5, anchor6]
    anchors = [a for a in anchors if isinstance(a, np.ndarray)]
    if len(anchors) > 1:
        lenx = len(anchors[0])
        if not all([len(a) == lenx for a in anchors]):
            raise ValueError('anchors do not match length')
        
    if verbose:
        print(label1)
        pprint(sq.get_percentiles(anchor1, digits=1))
        compare_anchor_to_gpt(anchor1)
        print('-')
        if anchor2 is not None and isinstance(anchor2, np.ndarray):
            print(label2)
            pprint(sq.get_percentiles(anchor2, digits=1))
            compare_anchor_to_gpt(anchor2)
            print('-')
        if anchor3 is not None and isinstance(anchor3, np.ndarray):
            print(label3)
            pprint(sq.get_percentiles(anchor3, digits=1))
            compare_anchor_to_gpt(anchor3)
            print('-')
        if anchor4 is not None and isinstance(anchor4, np.ndarray):
            print(label4)
            pprint(sq.get_percentiles(anchor4, digits=1))
            compare_anchor_to_gpt(anchor4)
            print('-')
        if anchor5 is not None and isinstance(anchor5, np.ndarray):
            print(label5)
            pprint(sq.get_percentiles(anchor5, digits=1))
            compare_anchor_to_gpt(anchor5)
            print('-')
        if anchor6 is not None and isinstance(anchor6, np.ndarray):
            print(label6)
            pprint(sq.get_percentiles(anchor6, digits=1))
            compare_anchor_to_gpt(anchor6)
            print('-')
        
        
    import matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)

    plt.hist(anchor1, bins=bins, alpha=alpha, label=label1, color='black', lw=0)

    if anchor2 is not None:
        if isinstance(anchor2, np.ndarray):
            plt.hist(anchor2, bins=bins, alpha=alpha, label=label2, color='limegreen', lw=0)
        else:
            plt.axvline(anchor2, label=label2, color='limegreen')

    if anchor3 is not None:
        if isinstance(anchor3, np.ndarray):
            plt.hist(anchor3, bins=bins, alpha=alpha, label=label3, color='red', lw=0)
        else:
            plt.axvline(anchor3, label=label3, color='red')

    if anchor4 is not None:
        if isinstance(anchor4, np.ndarray):
            plt.hist(anchor4, bins=bins, alpha=alpha, label=label4, color='blue', lw=0)
        else:
            plt.axvline(anchor4, label=label4, color='blue')

    if anchor5 is not None:
        if isinstance(anchor5, np.ndarray):
            plt.hist(anchor5, bins=bins, alpha=alpha, label=label5, color='orange', lw=0)
        else:
            plt.axvline(anchor5, label=label5, color='orange')

    if anchor6 is not None:
        if isinstance(anchor6, np.ndarray):
            plt.hist(anchor6, bins=bins, alpha=alpha, label=label6, color='brown', lw=0)
        else:
            plt.axvline(anchor6, label=label6, color='brown')
        
    if not disable_ci_lines:
        plt.axvline(np.mean(anchor1), label='{} (mean)'.format(label1), color='black')
        plt.axvline(np.percentile(anchor1, q=10), label='{} (10% CI)'.format(label1), color='black', linestyle='--')
        plt.axvline(np.percentile(anchor1, q=90), label='{} (90% CI)'.format(label1), color='black', linestyle='--')

        if anchor2 is not None and isinstance(anchor2, np.ndarray):
            plt.axvline(np.mean(anchor2), label='{} (mean)'.format(label2), color='green')
            plt.axvline(np.percentile(anchor2, q=10), label='{} (10% CI)'.format(label2), color='green', linestyle='--')
            plt.axvline(np.percentile(anchor2, q=90), label='{} (90% CI)'.format(label2), color='green', linestyle='--')

        if anchor3 is not None and isinstance(anchor3, np.ndarray):
            plt.axvline(np.mean(anchor3), label='{} (mean)'.format(label2), color='red')
            plt.axvline(np.percentile(anchor3, q=10), label='{} (10% CI)'.format(label3), color='red', linestyle='--')
            plt.axvline(np.percentile(anchor3, q=90), label='{} (90% CI)'.format(label3), color='red', linestyle='--')

        if anchor4 is not None and isinstance(anchor4, np.ndarray):
            plt.axvline(np.mean(anchor4), label='{} (mean)'.format(label2), color='blue')
            plt.axvline(np.percentile(anchor4, q=10), label='{} (10% CI)'.format(label4), color='blue', linestyle='--')
            plt.axvline(np.percentile(anchor4, q=90), label='{} (90% CI)'.format(label4), color='blue', linestyle='--')

        if anchor5 is not None and isinstance(anchor5, np.ndarray):
            plt.axvline(np.mean(anchor5), label='{} (mean)'.format(label2), color='orange')
            plt.axvline(np.percentile(anchor5, q=10), label='{} (10% CI)'.format(label5), color='orange', linestyle='--')
            plt.axvline(np.percentile(anchor5, q=90), label='{} (90% CI)'.format(label5), color='orange', linestyle='--')

        if anchor6 is not None and isinstance(anchor6, np.ndarray):
            plt.axvline(np.mean(anchor6), label='{} (mean)'.format(label2), color='brown')
            plt.axvline(np.percentile(anchor6, q=10), label='{} (10% CI)'.format(label6), color='brown', linestyle='--')
            plt.axvline(np.percentile(anchor6, q=90), label='{} (90% CI)'.format(label6), color='brown', linestyle='--')
    
    plt.xlim(xlim)
    plt.xlabel('log10 of total FLOP for training')
    plt.title('Distribution of total training compute needed')
    plt.legend()
    plt.show()
    return None
