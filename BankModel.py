def split_model(input_file, num_parts=8):
    with open(input_file, 'rb') as f:
        data = f.read()

    part_size = len(data) // num_parts
    for i in range(num_parts):
        part_data = data[i * part_size: (i + 1) * part_size]
        if i == num_parts - 1:  # Add the remaining bytes to the last part
            part_data += data[(i + 1) * part_size:]
        with open(f'part_{i + 1}.bin', 'wb') as part_file:
            part_file.write(part_data)
            
            
def GetBack_model(output_file, num_parts=8):
    with open(output_file, 'wb') as f_out:
        for i in range(num_parts):
            with open(f'part_{i + 1}.bin', 'rb') as part_file:
                f_out.write(part_file.read())            