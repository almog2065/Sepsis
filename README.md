# Sepsis-Label
Sepsis is a potentially life-threatening condition when the body’s response to an
infection causes widespread inflammation. It can lead to organ failure, septic
shock, and even death if not treated promptly and appropriately. Although
sepsis is currently the primary cause of death, its symptoms can be triggered
by various illnesses, making it difficult to diagnose.
For this task, we have been given files for 20,000 patients, where each file contains
medical and demographic information on each patient. Each row of the file corresponds to an
hourly measurement of these attributes. Our purpose is to train a model in
order to predict whether a patient is likely to suffer from sepsis in a 6 hours
period.
In our analysis, we take into account medical and demographic measures deemed
important for sepsis detection in the literature in addition to features found
significant in the data analysis presented in this document. Three algorithms
were used and compared for prediction purposes: random forest, neural network,
and XG-Boost.
## Code
The environment.yml file is for installation thhe relevant libraries.

The notebook contains all the code of the project.

The xgb_model.pkl is the trained model.

The prediction.py is the python file for prediction on the test set.


