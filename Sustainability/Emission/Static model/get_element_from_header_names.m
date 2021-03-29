function my_array_element = get_element_from_header_names( array_with_headers, row_name, col_name )
%GET_ELEMENT_FROM_HEADER_NAMES Get the element of an array which also
%contains row and column names, when indexed through those names.
%
% SYNTAX
%   [my_array_element] = get_element_from_header_names(
%   array_with_row_col_headers, row_name, col_name )
%
% DESCRIPTION
%   Returns the element of array_with_headers that corresponds to
%   the row with name row_name and the column with name row_name
%
% INPUTS
%   array_with_headers: Structure with three attributes:
%       - data: 2D array with elements
%       - colheaders: 1D cell array with the headers/names of the columns
%       - rowheaders: 1D cell array with the headers/names of the rows
%   row_name: name of the row that we want to index
%   col_name: name of the column that we want to index%   
%
% OUTPUT
%   my_array_element: the element corresponding to row row_name and column
%   col_name.
row_index = strcmp(array_with_headers.rowheaders, row_name);
col_index = strcmp(array_with_headers.colheaders, col_name);
if all(row_index == 0)
   ME = MException('RuntimeException:InvalidInput','The requested row header %s is not available.', row_name);
   throw(ME) 
end
if all(col_index == 0)
   ME = MException('RuntimeException:InvalidInput','The requested column header %s is not available.', col_name);
   throw(ME) 
end
% row_index = find(strcmp(array_with_headers.rowheaders, row_name));
% col_index = find(strcmp(array_with_headers.colheaders, col_name));
my_array_element = array_with_headers.data(row_index, col_index);
end
