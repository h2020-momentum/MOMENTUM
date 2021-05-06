# Induced demand model

Calculate new variable demand value, using the logit-based approach.

# Pre-requisites

[Matlab](https://www.mathworks.com/products/matlab.html), minimum version R2007b. Apart from the provided files, only standard functions are used.

# Description of the model files

The file `logit_based_variable_demand.m` implements a function with call signature

```
  [new_demand] = logit_based_variable_demand( utility, cal_param, base_utility, base_demand )
```
It returns the new demand as calculated through the logit-based formula for variable demand calibration.
One is free to choose the method for determining the calibration parameter.

# Citing this model
Vanherck, J. and Frederix, R. (2021). A framework for induced demand modelling. Manuscript in preperation.

# License
Distributed under the CC BY-NC-SA License

# Contact

-   Joren Vanherck, joren.vanherck@tmleuven.be
-   Rodric Frederix, rodric.frederix@tmleuven.be
