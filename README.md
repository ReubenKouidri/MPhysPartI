# Part I  MPhys 2022: Mutli-class ECG Classification

Code has been slightly modified in light of part II.

## The code
* <b>src</b>:
  * <b>datasets</b>:
    * <b>cpsc_data</b>:  
      Contains samples of data. Full dataset has not been included due to size, but can be found on the CPSC18
      website: http://2018.icbeb.org/Challenge.html
      * test100: 100 data samples
      * train300: 300 data samples (overlapping) 
      * reference300: labels for the 300 samples
  * <b>models</b>:
    * defaults.py: a default model for 2D CNN (other models not yet included)
    * components.py: for building the model - originally written for the NAS algo in part II to be used with '
      genetics' module.
  * <b>my_utils</b>:
    * config.json: run configurations in json format; to be used in conjunction with config.py.
    * config.py: works with .json file - see example provided. Originally designed for part II to initialise the NAS algorithm separately from argparse.
    * ksplit.py: allows for both custom number of splits and ratios of splits to be changed.
    * profiler.py: cprofile decorator
    * wavelets.py: for wavelet transforms used by 2D models
  * <b>distributed_run.py</b>: multi-GPU acceleration using DDP for train, test, score
  * <b>CPSCDataset.py</b>: two custom dataset classes that read, process and extract the data and are designed to work with instances of torch.utils.data.Dataloader
  * <b>main.py</b>: trial run to check set up, not the main loop that I used for the project
