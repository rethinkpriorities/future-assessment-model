import random
import numpy as np
from scipy import stats


RUNS = 1000
MAX_YEAR = 3000
USE_NUMBERS = 'Peter'
CREDIBILITY_INTERVAL = 0.9


if USE_NUMBERS == 'Ajeya':
    chance_ai_possible = 1              # What is the chance that Human-Level AI is possible in principle?
    chance_ai_abandoned = 0             # What is the chance that Human-Level AI is somehow abandoned?
    hlmi_flop_size = [13, 18]           # 90% CI - at how many inference FLOPS would we get transformative capabilities? In 10^X FLOPS for inference
    horizon = [0, 5]                    # 90% CI, this is the value of H for below
    train_inference_ratio = [14, 14]    # 90% CI, if an algorithm does Y FLOPS of inference it will take 10^H * 10^X * Y FLOPS to train it
    algo_dbl_yrs = [2, 3]               # 90% CI, every X years our algorithms get twice as efficient
    flop_dollar = [17, 18]              # 90% CI, currently we can purchase 10^X FLOPS for $1
    flop_halv_yrs = [2, 3]              # 90% CI, every X years each FLOP becomes half as expensive to purchase
    pay_hlmi = [8, 11]                  # 90% CI, $10^X in 2022 dollars is the maximum amount someone will pay for producing AI
elif USE_NUMBERS == 'Peter':
    chance_ai_possible = 0.97
    chance_ai_abandoned = 0.01
    hlmi_flop_size = [12, 21]
    horizon = [0, 5]
    train_inference_ratio = [13, 15]
    algo_dbl_yrs = [2, 4]
    flop_dollar = [17, 18]
    flop_halv_yrs = [2, 4]
    pay_hlmi = [8, 12]


def normal_sample(low, high, interval):
    if (low > high) or (high < low):
        raise ValueError
    if low == high:
        return low
    else:
        mu = (high + low) / 2
        cdf_value = 0.5 + 0.5 * interval
        normed_sigma = stats.norm.ppf(cdf_value)
        sigma = (high - mu) / normed_sigma
        return np.random.normal(mu, sigma)


def rand(var):
    return normal_sample(var[0], var[1], CREDIBILITY_INTERVAL)


def numerize(oom_num):
    oom_num = int(oom_num)
    ooms = ['thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion']

    if oom_num == 0:
        return 'one'
    elif oom_num == 1:
        return 'ten'
    elif oom_num == 2:
        return 'hundred'
    elif oom_num > 35:
        return numerize(oom_num - 33) + ' decillion'
    elif oom_num < 0:
        return 'negative ' + numerize(-oom_num)
    elif oom_num % 3 == 0:
        return ooms[(oom_num // 3) - 1]
    else:
        return str(10 ** (oom_num % 3)) + ' ' + ooms[(oom_num // 3) - 1]


hlmi_years = []

for r in range(RUNS):
    print('## RUN {}/{}'.format(r+1, RUNS))
    hlmi_created = False
    hlmi_year = 9999
    if random.random() > chance_ai_possible:
        print('In this run, human-level AI is not possible - even in principle.')
    elif random.random() < chance_ai_abandoned:
        print('In this run, the quest for human-level AI is somehow permanently abandoned.')
    else:
        hlmi_flop_size_ = rand(hlmi_flop_size); print('It takes {} FLOPS for inference to match a human brain.'.format(numerize(hlmi_flop_size_)))
        horizon_ = rand(horizon); train_inference_ratio_ = rand(train_inference_ratio); train_inference_ratio_ += horizon_
        print('An algorithm doing Y FLOPS of inference takes {} * Y FLOPS to train'.format(numerize(train_inference_ratio_)))
        algo_dbl_yrs_ = rand(algo_dbl_yrs); print('Every {} years algorithms get twice as efficient.'.format(np.round(algo_dbl_yrs_, 1)))
        flop_halv_yrs_ = rand(flop_halv_yrs); print('Every {} years FLOPs get twice as cheap.'.format(np.round(flop_halv_yrs_, 1)))
        pay_hlmi_ = rand(pay_hlmi); print('We are willing to pay {} 2022$ for HLMI.'.format(numerize(pay_hlmi_)))
        flop_dollar_ = rand(flop_dollar)
        print('---')
        for yr in range(2022, MAX_YEAR):
            if not hlmi_created:
                print('YEAR {}'.format(yr))
                algo_doublings = (yr - 2022) / algo_dbl_yrs_
                algo_doublings_log = np.log(algo_doublings)
                flop_halvings = (yr - 2022) / flop_halv_yrs_
                flop_halvings_log =  np.log(flop_halvings)
                if flop_halvings_log > 0:
                    flop_dollar__ = flop_dollar_ + flop_halvings_log
                else:
                    flop_dollar__ = flop_dollar_
                if algo_doublings_log > 0:
                    hlmi_flop_size__ = hlmi_flop_size_ - algo_doublings_log
                else:
                    hlmi_flop_size__ = hlmi_flop_size_
                print('FLOPS currently cost {} FLOPS per 2022$'.format(numerize(flop_dollar__)))
                flops_needed = hlmi_flop_size__ + train_inference_ratio_
                cost = hlmi_flop_size__ + train_inference_ratio_ - flop_dollar__
                print('So we need {} FLOPS to implement a human brain in 2022 and this will cost {} 2022$'.format(numerize(flops_needed), numerize(cost)))
                if cost <= pay_hlmi_:
                    print('/!\ HLMI CREATED!')
                    hlmi_created = True
                    hlmi_year = yr
                print('---')
    hlmi_years.append(hlmi_year)
    print('---')
    print('---')

percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
print(list(zip(percentiles, np.round(np.percentile(hlmi_years, percentiles)))))

# NOTE: Ajeya's numbers should output something very close to `[(10, 2030.0), (20, 2035.0),
# (30, 2040.0), (40, 2045.0), (50, 2053.0), (60, 2063.0), (70, 2073.0), (80, >2100.0)`

import pdb
pdb.set_trace()
