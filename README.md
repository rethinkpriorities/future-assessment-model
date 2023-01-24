# Future Assessment Model

This is an integrated assessment model to assess broad expectations of the outcomes of the next 100 years, by considering the development of transformative artificial intellignece and, various catastrophic and existential risks. It works by essentially implementing a finite state machine / a continuous-time Markov Chain.

**THIS IS UNFINISHED AND NOT READY FOR PRIMETIME. PLEASE DO NOT RE-SHARE. THIS MAY CHANGE DRAMATICALLY WITHOUT WARNING.** Note that there definitely needs to be and will be more documentation added - I know many aspects may currently be very hard to understand.

It takes place with the following notebooks in order:

* Start by installing pre-requisites: `pip3 install -r requirements.txt`

* **["(1) Anchors"]((1)%20Anchors.ipynb):** Analysis of different anchors for the compute needs for a transformative AI, following Ajeya Cotra (2020)'s ["Forecasting Transformative AI with Biological Anchors"](https://www.lesswrong.com/posts/KrJfoZzpSDpnrv9va/draft-report-on-ai-timelines)

* _optional_ **["(2) Cotra Timeline Models"]((2)%20Cotra%20Timeline%20Models.ipynb):** Replicate the AI timelines from ["Cotra (2020)"](https://www.lesswrong.com/posts/KrJfoZzpSDpnrv9va/draft-report-on-ai-timelines) and ["Cotra (2022)"](https://www.lesswrong.com/posts/AfH2oPHCApdKicM4m/two-year-update-on-my-personal-ai-timelines) in Python + [squigglepy](https://github.com/rethinkpriorities/squigglepy).

* **["(3A) Initial TAI Spend Model"]((3A)%20Initial%20TAI%20Spend%20Model.ipynb):** An attempt to model how much will be spent on the single largest AI training run in 2023. This informs the "initial pay" variable in the Cotra-like model.

* **["(3B) When TAI?"]((3B)%20When%20TAI%3F.ipynb):** Using the anchors from (1) and the spend from (3A), runs a modified Cotra-like model to re-estimate transformative AI arrival dates.

* _optional_ **["(3C) Timelines Sensitivity Analysis"]((3C)%20Timelines%20Sensitivity%20Analysis.ipynb):** A sensitivity analysis of how estimated TAI arrival changes based on different parameters in a Cotra-like model.

* _optional_ **["(3D) Short Timelines Sketch"]((3D)%20Short%20Timelines%20Sketch.ipynb):** A sketch of what someone who is expecting very short TAI timelines (e.g., >= 50% of arrival within five years) might input into a Cotra-like model to get short timelines results.

* **["(4) XRisk Model"]((4)%20XRisk%20Model.ipynb):** Using the TAI timelines from (3) and other estimates of risk, outputs an expectation of catastrophe (>= 10% death) risk, extinction risk, and existential risk over the next 100 years.
