
str = '@(x) 1+%d+%d+x';
varvar = {1, 4};

fprintf(str, varvar{:});
