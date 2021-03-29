%% Interprete configuration file and specify pollutants
cfg = process_config('emissions.cfg');
pollutants = { 'CO2'; 'CO'; 'NOX'; 'PM'; 'VOC'};

%% Read data regarding pollutant emission factors, co2 emission factors and
%% relative speed correction factors to the emission factors
speeds_ef_cf = read_csv_with_headers(cfg.file_speeds_ef_cf, '\t');
poll_ef = read_csv_with_headers(cfg.file_poll_ef, '\t');
co2_ef = read_csv_with_headers(cfg.file_CO2_ef, '\t');

%% Read data on links ID, length (in km), PCE and speed (in km/h)
file_link = fopen(cfg.file_link_info, 'rt');
link_info_bare = textscan(file_link, '%s\t%f\t%f\t%f', 'HeaderLines', 1, 'CollectOutput', 1);
fclose(file_link);
link_IDs = link_info_bare{1};
link_info = link_info_bare{2};

%% Construct speed correction factor functions for the different pollutants
correction_factor_f = construct_speed_correction_function(pollutants, speeds_ef_cf);

%% Calculate link emissions
% Extract the ef for the different pollutants
ef = cellfun(@(poll) get_emissions_factor(poll, cfg.country, cfg.year, poll_ef, co2_ef), pollutants);
% Initialize array that will hold link emissions
link_emissions = zeros(length(link_IDs), length(pollutants));
% Function that calculates link emissions based on length, pce and speed
calc_link_em_cal = @(len, pce, speed) calculate_link_emissions(len, pce,speed, ef, correction_factor_f)';

% Calculate link emissions for every link and totals
for i=1:length(link_IDs)
    link_emissions(i,:) = calc_link_em_cal(link_info(i,1), link_info(i,2), link_info(i,3));
end
tot_emissions = sum(link_emissions);

%% Make the output file
header_line = ['link' repmat('\t%s', 1, length(pollutants)) '\n'];
data_line = [ '%s' repmat('\t%.4f', 1, length(pollutants)) '\n'];

% Create output directory if necessary and open file
[filepath,name,ext] = fileparts(cfg.file_emissions_report);
if ~exist(filepath, 'dir')
    mkdir(filepath);
end
out_file = fopen(cfg.file_emissions_report,'w');

fprintf(out_file, '# Generated at %u-%u-%u %u:%u:%u\n', fix(clock));
fprintf(out_file, '# Country code: %s\n', cfg.country);
fprintf(out_file, '# Scenario year: %s\n', cfg.year);
fprintf(out_file, '#\n');
fprintf(out_file, '# Emission units: CO2 in g, others in mg\n#\n');
fprintf(out_file, '# Total emissions for network:\n');
for i=1:length(pollutants)
    fprintf(out_file, '#\t%s:\t%e\n', pollutants{i}, tot_emissions(i));
end
fprintf(out_file, '#\n#Emissions for each link:\n');

fprintf(out_file, header_line, pollutants{:});
for i=1:length(link_IDs)
    fprintf(out_file, data_line, link_IDs{i}, link_emissions(i,:));
end
fclose(out_file);

