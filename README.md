meanflow-jax/
├─ requirements.txt
├─ README.md
├─ configs/
│  └─ cifar10.yaml
└─ src/
   ├─ data/
   │  └─ cifar10.py
   ├─ core/
   │  ├─ schedules.py
   │  ├─ identity.py
   │  ├─ sample.py
   │  └─ utils.py
   ├─ models/
   │  ├─ blocks.py
   │  └─ meanflow_net.py
   └─ train.py


  # CRITICAL FIX: Sample r, t FIRST, then use t to create zt
  r, t = sample_r_t(rng_r_t, B)
  zt, v_t = linear_path(images, eps, t)