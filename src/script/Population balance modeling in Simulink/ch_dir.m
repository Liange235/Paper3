%path = 'C:\Users\t1714\Desktop\Academic\Coding_Files\GP building polymers\Population balance modeling in Simulink\Population balance modeling in Simulink\PCSS_ZIP'
function gen = ch_dir(path)
    gen = path;
    cd(path);
    load("PCSS_ex_2_data.mat");
end