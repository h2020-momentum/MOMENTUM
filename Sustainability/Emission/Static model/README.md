# Emission model

Calculate emissions for pollutants CO, CO2, NOx, PM and VOC. The model accounts for varying fleet compositions over EU countries in different years (2016 to 2050). Moreover, it corrects for the speed-dependence of emissions on a per-link basis.

The output consists of emission estimations for each link and cumulative emissions over the entire network.

# Pre-requisites

[Matlab](https://www.mathworks.com/products/matlab.html), minimum version R2007b. Apart from the provided files, only standard functions are used.

# Description of the model files

The model consists of Matlab files (.m files), input data files (.csv files) and a configuration file (.cfg file).

In the configuration file `emissions.cfg`, the user can specify the country and year for which the emissions should calculated. The supported country codes can be found in `country_codes.xlsx`. Furthermore, it allows to specify the names and locations of the input and output files (standard: `./out/pollutant_report.csv`). The input files needed for the model are (formats are clear from the provided files):

-   CO2_ef_file: $CO_2$ emission factors for the years and countries that you want to model (`dat/CO2_tailpipe_emission_per_year-country.csv` provided for EU countries between 2016 and 2050);
-   pollutants_ef_file: $CO$, $NOX$, $PM$ and $VOC$ emission factors for the years and countries that you want to model (`dat/Emission_factors_per_year-country-pollutant.csv` provided for EU countries between 2016 and 2050);
-   speeds_correction_factors_file: Correction factors to the emission factors for the different pollutants at different speeds (`dat/speed_correction_factors.csv` provided);
-   link_info_file: File that provide info from network. This file should be generated for the specific network that you want to model (`dat/links_info.csv` provided solely as reference for the correct data format). For each link, this file should provide:
    -   the link id (positive integer);
    -   the link length (in km);
    -   the link flow (passenger car equivalent, pce);
    -   the speed (in km/h).

The provided Matlab files are (relationship between files is shown in figure):

-   `calculate_emissions.m`: The main file of the modelling script. This is the file that should be executed.
-   `process_config.m`: Function that processes the configuration file.
-   `read_csv_with_headers.m`: Function that reads a csv files, including its column/row headers. This function is used to read the speeds_correction_factors_file, pollutants_ef_file and CO2_ef_file.
-   `get_element_from_header_names.m`: Helper function to get the element of an array which also contains row and column names, when indexed through those names.
-   `get_emissions_factors.m`: Function that gets the emission factor of a pollutant for a given country and year.
-   `construct_speed_correction_function.m`: Function that constructs a function that interpolates the speed correction factors for the pollutant emission factor to intermediate speeds.
-   `calculate_link_emissions.m`: Function that calculates the emissions for a link.

![call diagram](call_diagram.png)

# Citing this model
Vanherck, J., Vanherle, K. and Frederix, R. (2021). Speed-dependent emission factors for traffic assignment models within the EU. Manuscript in preperation.

# License
Distributed under the CC BY-NC-SA License

# Contact

-   Joren Vanherck, joren.vanherck@tmleuven.be
-   Kris Vanherle, kris.vanherle@tmleuven.be
-   Rodric Frederix, rodric.frederix@tmleuven.be
