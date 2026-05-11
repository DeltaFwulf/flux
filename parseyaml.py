"""Test script to try parsing a yaml file."""
import yaml



def parse_yaml():
    """Test function for loading data from yaml file."""

    with open('testrun.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    print(data['meshes']['mesh0']['lines'][0])



parse_yaml()
