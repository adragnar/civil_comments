import os
from os.path import join
import argparse
import shutil

def save_reproducability(cdir, rdir):
    '''Save all code needed to reproduce given experiments
    :param rdir: Dirname of 'causal_discovery' folder on server
    return save in 'code' folder '''

    code_folder = join(rdir, 'code')
    os.mkdir(code_folder)

    for fname in os.listdir(cdir):
        #First deal with source files
        if os.path.isfile(join(cdir, fname)):
            shutil.copy(join(cdir, fname), join(code_folder, fname))

        #Now deal with launchfiles
        if fname == 'launchfiles':
            shutil.copytree(join(cdir, fname), join(code_folder, fname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument("c_dir", type=str)
    parser.add_argument("r_dir", type=str)
    args = parser.parse_args()
    save_reproducability(args.c_dir, args.r_dir)  
