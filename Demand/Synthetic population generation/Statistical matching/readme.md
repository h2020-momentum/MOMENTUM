# Statistical matching
This model is intended to enrich the synthesised population, generated using the PopGen tool, with additional information. For example, a sample from the synthetic population can be combined with data from the household travel surveys for attributes (e.g. if a person holds a driver´s license) that may not be available both in the census and survey samples to include them in the initial population synthesis. The process involves the definition of mutual sociodemographic attributes in the synthetic population and sample surveys (e.g. age, gender, income class), assuming sufficient correlation with the unilateral attributes to attach to the synthetic persons.
CSV files of the synthetic population, the computed weights for each household as well as sample data (e.g., from household travel surveys) are taken as input and the output is also a CSV file, containing the synthetic population with the additional attributes. The method is adopted from (*Hörl, S. and Balac, M., 2020. Reproducible scenarios for agent-based transport simulation: A case study for Paris and Île-de-France.*)
Pre-requisites
This model will be coded in Python language. More details regarding any special packages that may be required will be provided when the model will be completed. For details about Python, kindly check https://www.python.org/.

# Description of the model files
1.	*InputSyntheticPopulation.csv* (in progress) – File with synthesised population observations derived from the PopGen tool
2.	*Weights.csv* (in progress) – File with the household weights computed from PopGen
3.	*InputSampleSurvey.csv* (in progress) – File with household travel survey observations 
4.	StatisticalMatching.py (in progress) – Main model script
# Citing this model
In progress.

# License

# Contact
Athina Tympakianaki, athina.tympakianaki@aimsun.com
