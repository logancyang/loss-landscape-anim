import numpy as np
from loss_landscape_anim import loss_landscape_anim

u_gen = np.random.normal(size=303)
u = u_gen / np.linalg.norm(u_gen)
v_gen = np.random.normal(size=303)
v = v_gen / np.linalg.norm(v_gen)

loss_landscape_anim(n_epochs=300, load_model=False)
# loss_landscape_anim(n_epochs=300, load_model=False, custom_directions=(u, v))
