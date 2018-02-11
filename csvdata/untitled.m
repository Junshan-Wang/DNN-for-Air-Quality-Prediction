for i = 1:35
    data = air_bj{i, 1};
    filename = ['data' num2str(i) '.csv'];
    disp(filename) ;
    csvwrite(filename, data);
end