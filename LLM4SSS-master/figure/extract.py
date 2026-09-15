selected_model = "TimeMixer"
selected_dataset = "sald"

log_path = f"example/{selected_model}/{selected_dataset}/log/example.log"
save_path = f"figure/error/{selected_model}_{selected_dataset}.txt"

first_extracted = False

with open(log_path, "r") as log_file:
    with open(save_path, "w") as output_file:
        for line in log_file:
            if "validate trans_err: " in line:
                trans_err_value = line.split("validate trans_err: ")[1].strip()
                
                if not first_extracted:
                        first_extracted = True
                        continue
                
                output_file.write(trans_err_value + "\n")