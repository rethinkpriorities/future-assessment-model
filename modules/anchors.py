def get_anchor_gpt_benchmark_pct(anchor, target):
    out = round(np.mean([a <= target for a in anchor]) * 100, 1)
    return '<0.1%' if out == 0 else str(out) + '%'


def compare_anchor_to_gpt(anchor):
    print('-')
    print('GPT2 (~21 log FLOP) can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, 21)))
    print('GPT3 (~23 log FLOP) can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, 23)))
    print('GPT4 (~25 log FLOP) can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, 25)))
    print('GPT5 (~27 log FLOP) can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, 27)))
    print('GPT6 (~28 log FLOP) can do it: {}'.format(get_anchor_gpt_benchmark_pct(anchor, 28)))


def tai_log_flop_needs(brain, efficiency, transformative_vs_human, horizon_length, scaling_exponent,
                       flops_per_param_per_sec, params=None, ref_params=11.2, ref_params_samples=12,
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
    f = f + ~sq.norm(1,3) if f < 27 and random.random() > 0.3 else f
    f = f + ~sq.norm(1,3) if f < 26 and random.random() > 0.2 else f
    f = f + ~sq.norm(1,3) if f < 25 and random.random() > 0.1 else f
    f = f + ~sq.norm(1,3) if f < 24 else f
    f = 24 if f < 24 else f
    return f


@np.vectorize
def peter_bayes_update_against_low_flop(f):
    f = f + ~sq.lognorm(1,3) if f < 26 else f
    f = f + ~sq.lognorm(1,4) if f < 25 else f
    f = f + ~sq.lognorm(1,5) if f < 24 else f
    f = f + ~sq.lognorm(2,6) if f < 23 else f
    f = 23 if f < 23 else f
    return f


def cotra_anchor(horizon_length, bayes_update=cotra_bayes_update_against_low_flop, chinchilla=False):
    return tai_log_flop_needs(brain=sq.lognorm(11,19.5),
                              efficiency=1,
                              transformative_vs_human=sq.norm(-2,2),
                              horizon_length=horizon_length,
                              scaling_exponent=sq.norm(0.5, 1.5 if chinchilla else 1.1),
                              flops_per_param_per_sec=sq.norm(1,2),
                              bayes_update=bayes_update)


def plot_anchors(anchor1=None, anchor2=None, anchor3=None, bins=100, alpha=0.6, label1=None, label2=None, label3=None,
                verbose=True, xlim=[20, 75], figsize=(10,8)):
    if label1 is None:
        label1 = 'Anchor1'
    if label2 is None:
        label2 = 'Anchor2'
    if label3 is None:
        label3 = 'Anchor3'
        
    if anchor1 is None:
        raise ValueError
        
    if anchor2 is None and anchor3 is not None:
        raise ValueError('{} defined without defining {}'.foramt(label3, label2))

    if anchor2 is not None and len(anchor1) != len(anchor2):
        raise ValueError('{} and {} do not match length'.format(label1, label2))

    if anchor3 is not None and len(anchor2) != len(anchor3):
        raise ValueError('{} and {} do not match length'.format(label2, label3))
        
    if verbose:
        print(label1)
        pprint(sq.get_percentiles(anchor1, digits=1))
        compare_anchor_to_gpt(anchor1)
        print('-')
        if anchor2 is not None:
            print(label2)
            pprint(sq.get_percentiles(anchor2, digits=1))
            compare_anchor_to_gpt(anchor2)
            print('-')
        if anchor3 is not None:
            print(label3)
            pprint(sq.get_percentiles(anchor3, digits=1))
            compare_anchor_to_gpt(anchor3)
            print('-')
        
    if xlim:
        anchor1 = [xlim[0] if a < xlim[0] else a for a in anchor1]
        anchor1 = [xlim[1] if a > xlim[1] else a for a in anchor1]
        if anchor2 is not None:
            anchor2 = [xlim[0] if a < xlim[0] else a for a in anchor2]
            anchor2 = [xlim[1] if a > xlim[1] else a for a in anchor2]
        if anchor3 is not None:
            anchor3 = [xlim[0] if a < xlim[0] else a for a in anchor3]
            anchor3 = [xlim[1] if a > xlim[1] else a for a in anchor3]
        
    plt.figure(figsize=figsize)
    plt.hist(anchor1, bins=bins, alpha=alpha, label=label1, color='black', lw=0)
    if anchor2 is not None:
        plt.hist(anchor2, bins=bins, alpha=alpha, label=label2, color='limegreen', lw=0)
    if anchor3 is not None:
        plt.hist(anchor3, bins=bins, alpha=alpha, label=label3, color='red', lw=0)
        
    plt.axvline(np.mean(anchor1), label='{} (mean)'.format(label1), color='black')
    if anchor2 is not None:
        plt.axvline(np.mean(anchor2), label='{} (mean)'.format(label2), color='green')
    if anchor3 is not None:
        plt.axvline(np.mean(anchor3), label='{} (mean)'.format(label2), color='red')
    
    plt.axvline(np.percentile(anchor1, q=10), label='{} (10% CI)'.format(label1), color='black', linestyle='--')
    if anchor2 is not None:
        plt.axvline(np.percentile(anchor2, q=10), label='{} (10% CI)'.format(label2), color='green', linestyle='--')
    if anchor3 is not None:
        plt.axvline(np.percentile(anchor3, q=10), label='{} (10% CI)'.format(label3), color='red', linestyle='--')
    
    plt.axvline(np.percentile(anchor1, q=90), label='{} (90% CI)'.format(label1), color='black', linestyle='--')
    if anchor2 is not None:
        plt.axvline(np.percentile(anchor2, q=90), label='{} (90% CI)'.format(label2), color='green', linestyle='--')
    if anchor3 is not None:
        plt.axvline(np.percentile(anchor3, q=90), label='{} (90% CI)'.format(label3), color='red', linestyle='--')
        
    plt.xlim(xlim)
    plt.legend()
    plt.show()
    return None
