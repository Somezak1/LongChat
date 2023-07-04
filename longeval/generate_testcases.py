from pathlib import Path
from utils import generate_topics_testcases, generate_lines_testcases, retrieve_cmd_args

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", 
    #                     type=str, required=True, 
    #                     help="directory to put generated testcases")
    # parser.add_argument("--task", 
    #                     type=str, required=True, 
    #                     help="Which evaluation testcases to generate. currently support [topics, lines]")

    # args = parser.parse_args()
    cfgs = retrieve_cmd_args()

    output_dir = Path(cfgs["output_dir"]) / Path(cfgs["task"]) / Path("testcases/")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if cfgs["task"] == "topics":
        generate_topics_testcases(cfgs, output_dir)
    else:
        generate_lines_testcases(cfgs, output_dir)