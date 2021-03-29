function csv_content = read_csv_with_headers( csv_location, sep_char )
% READ_CSV_WITH_HEADERS Read a csv file, including its column/row headers
%
% SYNTAX
%   [csv_content] = read_csv_with_headers( csv_location, sep_char ) 
%
% DESCRIPTION
%   Returns the content of the csv at csv_location.
%   The data, colheaders and rowheaders are stored in the attributes
%   with the same name in csv_content.
%   Note that the first row should contain the column headers and the first
%   column should contain the row headers
%
%INPUTS
%   csv_location: csv file to be read.
%   sep_char: The separation character in the csv file
%
% OUTPUT
%   csv_content: The content of the csv that was read as a structure. It
%   has three attributes:
%       - data: the data in the csv
%       - colheaders: the column headers as character cells
%       - rowheaders: the row headers as character cells
csv_content = importdata(csv_location, sep_char, 1);
csv_content.colheaders = csv_content.textdata(1,2:end)';
csv_content.rowheaders= csv_content.textdata(2:end,1);
csv_content = rmfield(csv_content, 'textdata');
end


