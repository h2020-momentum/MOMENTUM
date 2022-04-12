# Statistical matching
This model is intended to enrich the synthesised population, generated using the PopGen tool, with additional information. For example, a sample from the synthetic population can be combined with data from the household travel surveys for attributes (e.g. if a person holds a driver´s license) that may not be available both in the census and survey samples to include them in the initial population synthesis. The process involves the definition of mutual sociodemographic attributes in the synthetic population and sample surveys (e.g. age, gender, income class), assuming sufficient correlation with the unilateral attributes to attach to the synthetic persons.
CSV files of the synthetic population, the computed weights for each household as well as sample data (e.g., from household travel surveys) are taken as input and the output is also a CSV file, containing the synthetic population with the additional attributes. The method is adopted from (*Hörl, S. and Balac, M., 2020. Reproducible scenarios for agent-based transport simulation: A case study for Paris and Île-de-France.*)

# Pre-requisites
This model is coded in Python language and works with Python >= 3.7.6. The following packages are required:

- scikit-learn >= 1.0.2
- numpy >= 1.21.5
- pandas >= 1.4.2

For details about Python, kindly check https://www.python.org/.

# Description of the model files
1.	extend_dataset_with_resampling.py – Main model script
2.	data_extension.py – Python file with useful functions
3.	*survey.csv* – CSV file with the training dataset from household travel survey observations 
4.	*agents.csv* – CSV file with the dataset to extend

# How to use it

extend_dataset_with_resampling.py is the main script (the one to run). Before running it:
- Edit from line 9 to 19 to set the folder with survey.csv and agents.csv files,
the output file, the common attributes in both datasets and the target attributes (the ones to estimate for the target
population).
- Edit "M" (line 16) which corresponds to the minimum number of source observations from where to draw the common attributes. 
- Select the columns that correspond to the select mutual attributes from the survey file or provide them in a list (lines 17-18)
- Define the target (new) attributes to be added to the synthetic population (line 19) based on the names of the attributes as provided in the survey file.


# Citing this model
*Hörl, S. and Balac, M., 2020. Reproducible scenarios for agent-based transport simulation: A case study for Paris and Île-de-France.*

# License
Distributed under the MIT License.

# Contact
Athina Tympakianaki, athina.tympakianaki@aimsun.com

Ferran Torrent, ferran.torrent@aimsun.com
