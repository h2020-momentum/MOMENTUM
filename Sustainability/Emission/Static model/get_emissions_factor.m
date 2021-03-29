function emission_factor = get_emissions_factors( pollutant, country, year, pollutants_ef, co2_ef )
%GET_EMISSIONS_FACTORS gets the emission factor of a pollutant for a given
%country and year
%
% SYNTAX
%   [emission_factor] = get_emissions_factors( pollutant, country, year,
%   pollutants_ef, co2_ef )
%
% DESCRIPTION
%   Returns the emission factor of the requested pollutant for a given
%   country and year, based on the data in pollutants_ef and cot2_ef. If
%   the combination of pollutant, country and year is not present in the
%   data, an exception is thrown.
%
% INPUTS
%   - pollutant: String specifying pollutant. Supported choices are 'CO',
%       'NOX', 'PM', 'VOC' and 'CO2'
%   - country: String specifying country. Only Vensim country codes
%       (see file 201204_VENSIM_country_codes.xlsx) are supported.
%   - year: String or integer specifying year
%   - pollutants_ef: table with emission factors for pollutants other than
%       CO2
%   - co2_ef: table with emission factors for CO2
%
% OUTPUT
%   emission_factor: the requested emission factor, as a double
if isinteger(year)
    year = int2str(year);
end

try
    if strcmp(pollutant, 'CO2')
        row_name = ['[' country ']'];
        emission_factor = get_element_from_header_names(co2_ef, row_name, year);
    else
        row_name = ['[' country ',' pollutant ']'];
        emission_factor = get_element_from_header_names(pollutants_ef, row_name, year);
    end
catch ME1
    if strcmp(ME1.identifier, 'RuntimeException:InvalidInput')
        ME = MException('RuntimeException:InvalidInput', ...
        'The combination of pollutant %s, country %s and year %s is not available in the data.', ...
        pollutant, country, year);
        throw(ME)
    else
        rethrow(ME1)
    end
end


