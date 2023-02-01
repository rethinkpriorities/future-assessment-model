My initial impression is that I find it a bit hard to read. It's possible that a large part of this comes from me not actually knowing python. If this model was mature, I might suggest:

- Making a diagraph of its structure in excalidraw
- For each important function, have a comment explaining what it does. 

Re: Aleya's model:

- Really not sure of how you are computing the Bayesian update
- I'm not sure whether you've seen <https://nostalgebraist.tumblr.com/post/693718279721730048/on-bio-anchors>, which claims that one of the more important assumptions in the model is the continuance of Moore's law, which isn't assured
- Re: the evolution anchor, I think it underweighs the time because of reasons seen here: <https://nunosempere.com/blog/2022/08/10/evolutionary-anchor/>
- "params" is a terrible variable name, maybe `model_parameters_needed`
- `transformative_vs_human=sq.norm(-2,2)` why?
  - In particular, an AI which was as capable as 100 humans is/isn't transformative depending on the training cost, no?
- also, not really a fan of normals
- what is `sq.dist_fn` and how is it used to represent a Bayes update?
- maybe consider labelling the x and y axes
- I would be curious where you are getting the numbers in `environment_adjustment = sq.mixture([[0.2, sq.lognorm(1,5)], [0.8, 0]])` from.
  - also, wow, only a 20% probability of [what I think is a distribution centered around 1 oom more], brutal.
- As this becomes more mature, I'd advertise somewhere prominent that you'd need python3.9 to run it.
  - also, in linux, I somehow needed to reinstall the psutil package
- `dump_cache_file='caches/cotra_2020` is hardcoded, man.
- the meta-anchor name makes no sense, should be called "personal judgment anchor"
  - the meta-anchor should be some mixture of object level anchors.
- Not sure what is happening in the backderive section.
- The model gives an error if there is no caches directory. I also don't see a damage in also saving the cache
- Would probably be good to warn before the backderivation that it's going to take long.
