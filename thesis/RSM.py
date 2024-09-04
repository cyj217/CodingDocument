import random
import pandas as pd

def generate_response_surface_samples(parameter_ranges, sample_size):
    samples = []
    for _ in range(sample_size):
        sample = []
        for i, param_range in enumerate(parameter_ranges):
            if i in [4, 5]:  # Check if parameter index is for n_down or n_up
                param_value = random.randint(param_range[0], param_range[1])
            else:
                param_value = round(random.uniform(param_range[0], param_range[1]), 2)
            sample.append(param_value)
        samples.append(sample)
    return samples

parameters = ["x_down", "x_up", "z_down", "z_up", "n_down", "n_up", "p_down", "p_up"]
parameter_ranges = [[20, 40], [20, 40], [5, 12], [5, 12], [3, 6], [3, 6], [400, 800], [400, 800]]
sample_size = 100

response_surface_samples = generate_response_surface_samples(parameter_ranges, sample_size)

# Create a DataFrame from the samples
df = pd.DataFrame(response_surface_samples, columns=parameters)

# Export DataFrame to a txt file
df.to_csv('parameter_samples.txt', sep='\t', index=False, float_format='%.2f')

print("Parameter samples exported to parameter_samples.txt")