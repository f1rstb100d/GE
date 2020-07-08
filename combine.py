file_obj = open("relink0601.txt",'r')
all_lines = file_obj.readlines()
file_obj.close()

file_write_obj = open("combinelink.txt", 'a')
for var in all_lines:
    var = var.replace('\n',' 10\n')
    file_write_obj.writelines(var)
    # file_write_obj.write('\n')
file_write_obj.close()
