# kaggle G2Net Detecting Continuous Gravitational Waves 10th Place Solution

----------------------------------------------------------------

**I([anonamename](https://www.kaggle.com/anonamename)) did all solutions by myself. I did not get help from my teammates.**

**Due to the lack of time to organize the code, the code used in the experiment is also included in this repository.**

------------------------------------------------------------------------------------------------------

competition: https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves

solution summary: https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/376052

## Hardware

- Ubuntu 18.04.6 LTS
- Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz / AMD Ryzen 9 5900X 12-Core Processor
- NVIDIA Teslas V100 32G / NVIDIA GeForce RTX 3090
- Memory 64GB

## Software

Used the following docker image for the execution environment.

- Data generation by PyFstat:
  
  - `docker pull ghcr.io/pyfstat/pyfstat/pyfstat:latest` 
  
  - Python packages are detailed separately in [./gen_data/requirements.txt](./gen_data/requirements.txt)

- Train, Prediction: 
  
  - `docker pull sinpcw/pytorch:1.11.0`
  
  - Python packages are detailed separately in [./train/requirements.txt](./train/requirements.txt)

## Data generation by PyFstat

- Run all notebooks in the [./gen_data](./gen_data)
  
  - Requires approximately 2TB of free space in storage

## Training, Prediction

- Run all the notebooks in [./train](./train) all notebooks in the following order.
- Iterate the process of creating pseudo-labels for the test set in the LB improved model and add them to the training data.

```bash
# Train without pseudo-labels
lb0.747_3090_kqi_ex066_public_add_ex003_2-005-01_005iso-01_006_fold01234_b5_ap_largekernel_p-robustscaler_ts-ma.ipynb

# Train with pseudo-labels
lb0.755_3090_kqi_ex073_pseudo_add_ex003_2-005-01_005iso-01_006_b5_ap_add_pseudo_EX007_800.ipynb
lb0.759_kqi_3090_ex073_pseudo_add_ex003_2-005-01_005iso-01_006_b4_ap_n_fold20.ipynb
lb0.762_kqi_3090_ex075_pseudo_multioutput_freq_b5_ap.ipynb
lb0.771_kqi_3090_ex075_v2_pseudo_multioutput_freq_b5_ap_freq_div_n50_TTA.ipynb
lb0.772_kqi_3090_ex075_v2_tta_pseudo_hvflip_lb0771_09501_simall.ipynb
lb0.773_kqi_3090_ex075_v2_tta_pseudo_hvflip_lb0771_1000_simall.ipynb
lb0.774_kqi_3090_ex075_v2_tta_v2_pseudo_hvflip_lb0771_1000_simall_100ep.ipynb
lb0.775_kqi_3090_ex075_v2_tta_v4_norm_hvflip_lb0771_pseudo_th_5_10_100ep.ipynb
lb0.776_3090_kqi_ex075_v2_tta_v4_norm_stride12_lb0771_pseudo_th_5_10_100ep_addD_EX007.ipynb
lb0.778_3090_kqi_ex075_v2_tta_v4_norm_stride12_lb0771_pseudo_th_5_10_100ep_addD_EX007_2.ipynb
lb0.779_kqi_3090_ex075_v2_stride12_norm_lb0771_pseudo_th_5_10_addD_EX007_2_GaussNoise.ipynb
lb0.780_kqi_3090_ex075_v2_stride12_norm_lb0771_pseudo_th_5_10_addD_EX007_3_GN.ipynb
```

## Submit

- Rank average of created submit.csv.
  
  - best submit: https://www.kaggle.com/code/anonamename/g2net2-sub-avg/notebook?scriptVersionId=115254793
