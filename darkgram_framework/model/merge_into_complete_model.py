import os

def split_model(file_path, part_size_mb=90):
    part_size = part_size_mb * 1024 * 1024  
    output_dir = os.getcwd()  

    with open(file_path, 'rb') as f:
        data = f.read()

    total_size = len(data)
    num_parts = (total_size + part_size - 1) // part_size

    for i in range(num_parts):
        start = i * part_size
        end = min(start + part_size, total_size)
        part_data = data[start:end]

        part_file_name = os.path.join(output_dir, f'model_part_{i+1:03d}.part')
        with open(part_file_name, 'wb') as part_file:
            part_file.write(part_data)

    print(f"Model split into {num_parts} parts and saved in '{output_dir}'.")

def merge_model(output_file):
    input_dir = os.getcwd()  
    part_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.part')]
    )

    with open(output_file, 'wb') as output:
        for part_file in part_files:
            with open(part_file, 'rb') as f:
                output.write(f.read())

    print(f"Parts merged into '{output_file}'.")

# Split
#split_model('model.safetensors')

# Merge
merge_model('model.safetensors')
