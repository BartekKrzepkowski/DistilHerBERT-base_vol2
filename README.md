# DistilHerBERT-base_vol2

You can run distillation via

accelerate launch distil_trainer_entropy.py

command.

However, you must first set the accelerate config in line with e.g.
https://jarvislabs.ai/blogs/accelerate/?fbclid=IwAR2-Plufxv7UtjsFq-Yqb7TzEd4keaBQyzb3w3xO8G2eMd7ihARaTCPJnXs

You should also change the path to the tokenized dataset inside this script in the "run" method (sorry).

I can share the tokenized CCsub (as long as I have it saved somewhere by then).
