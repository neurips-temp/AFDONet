# AFDONet
This is the temporary codebase for AFDONet, a novel neural PDE solver for nonlinear PDEs on manifolds. The associated paper submitted to [NeurIPS 2025] for review is titled "AFDONet: Enhancing Partial Differential Equation Learning on Manifolds via Adaptive Fourier Decomposition".

#### 1. Specification of dependencies

If you are using Python, this means providing a `requirements.txt` file (if using `pip` and `virtualenv`), providing `environment.yml` file (if using anaconda), or a `setup.py` if your code is a library. 

It is good practice to provide a section in your README.md that explains how to install these dependencies. Assume minimal background knowledge and be clear and comprehensive - if users cannot set up your dependencies they are likely to give up on the rest of your code as well. 

If you wish to provide whole reproducible environments, you might want to consider using Docker and upload a Docker image of your environment into Dockerhub. 

#### 2. Training code

Your code should have a training script that can be used to obtain the principal results stated in the paper. This means you should include hyperparameters and any tricks that were used in the process of getting your results. To maximize usefulness, ideally this code should be written with extensibility in mind: what if your user wants to use the same training script on their own dataset?

You can provide a documented command line wrapper such as `train.py` to serve as a useful entry point for your users. 

#### 3. Evaluation code

Model evaluation and experiments often depend on subtle details that are not always possible to explain in the paper. This is why including the exact code you used to evaluate or run experiments is helpful to give a complete description of the procedure. In turn, this helps the user to trust, understand and build on your research.

You can provide a documented command line wrapper such as `eval.py` to serve as a useful entry point for your users.

#### 4. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

#### 5. README file includes table of results accompanied by precise command to run to produce those results

Adding a table of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. In some cases, the main result of a paper is a Figure, but that might be more difficult for users to understand without reading the paper. 

You can further help the user understand and contextualize your results by linking back to the full leaderboard that has up-to-date results from other papers. There are [multiple leaderboard services](#results-leaderboards) where this information is stored.  
