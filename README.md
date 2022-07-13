## Machine Learning and Pattern Recognition Project
Authors: Michele Cerra, Giuseppe Stracquadanio

### Project Structure for the Delivery
* _main.py_ script: testing performances of all the models on the **val.** set (K-fold CV)
* _evalMain.py_ script: testing performances of all the models on the **eval.** set
* _calibrationFusionMain.py_: performing score calibration and fusion for the best models on the val. set
* _finalMain.py_: testing the optimal model(s) on the evaluation set 

### Pipeline Abstraction
For better managing all the models and all the steps, we employed the _abstraction_ of a pipeline. Each model (_classifiers.py_) or pre-processing step (_preProc.py_) is a pipeline stage, which can be specified when creating a new pipeline. Then, the pipeline can be received as input by a CrossValidator object, in order to execute all the steps while using the K-fold protocol. 