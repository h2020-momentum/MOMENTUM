function  cfg  = process_config( file_name )
% PROCESS_CONFIG Process the configuration file
%
% SYNTAX
%   cfg = process_config( file_name )
%
% DESCRIPTION
%   Processes content of configuration file. The extracted variables are
%   returned through a structure
%
% INPUTS
%   file_name: name/location of the configuration file.
%
% OUTPUT
%   cfg containing info that was presen in the configuration file.

f = fopen(file_name, 'rt');
inconfig = textscan(f, '%s %*[=] %[^\n]', 'CommentStyle', '#');
fclose(f);

cfg = {};
cfg.country = get_value_from_config_cells('country_code', inconfig);
cfg.year = get_value_from_config_cells('year', inconfig);
cfg.file_speeds_ef_cf = get_value_from_config_cells('speeds_correction_factors_file', inconfig);
cfg.file_poll_ef = get_value_from_config_cells('pollutants_ef_file', inconfig);
cfg.file_CO2_ef = get_value_from_config_cells('CO2_ef_file', inconfig);
cfg.file_link_info = get_value_from_config_cells('link_info_file', inconfig);
cfg.file_emissions_report = get_value_from_config_cells('output_report_file', inconfig);
cfg.country = get_value_from_config_cells('country_code', inconfig);
cfg.year = get_value_from_config_cells('year', inconfig);



function var_value = get_value_from_config_cells(var_name, config_cells)
% GET_VALUE_FROM_CONFIG_CELLS Extract a parameter that was read in from the
% configuration file
%
% SYNTAX
%   var_value = get_value_from_config_cells(var_name, config_cells)
%
% DESCRIPTION
%   The (variable, value) pairs are extracted into two separate cells from
%   the config file. The cells need to be matched to extract a variable
%   with its value. Given a variable name, this function will extract its
%   value.
%
% INPUTS
%   var_name: Name of variable (as givne in the config file) to be
%       extracted.
%   config_cells: Cell structure obtained from reading in the config file.
%
% OUTPUT
%   value corresponding to the variable, as given in the config file.

index = find(strcmp([config_cells{1}], var_name));

if length(index) == 1
    var_value = cell2mat(config_cells{2}(index));
else
    error_type = 'RuntimeException:InvalidInput';
    if isempty(index)
        error_msg = 'The required parameter %s is not specified in the configuration file.';
    else
        error_msg = 'The required parameter %s is specified multiple times in the configuration file.';
    end
    ME = MException(error_type,error_msg, var_name);
    throw(ME)
end